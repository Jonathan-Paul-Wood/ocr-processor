from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import threading
import time

import pytest

from ocr_processor.app import create_app
from ocr_processor.errors import ServiceError
from ocr_processor.jobs import JobStore
from test_transactions import CONTEXT, SCHEMA, transaction_ranker


PAYLOAD = b'Date,Description,Amount\n2026-11-15,Market,12.34\n'


def wait_ready(store, identity):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = store.snapshot(identity)
        if result['ready']:
            return result
        threading.Event().wait(.01)
    pytest.fail('Job did not finish')


def manifest(payload=PAYLOAD):
    return [dict(job_id=hashlib.sha256(payload).hexdigest(), filename='rows.csv')]


def test_jobs_are_deduplicated_until_all_consumers_ack(settings):
    gate = threading.Event()
    calls = []
    def runner(directory, cancelled):
        calls.append(directory)
        assert gate.wait(5)
        return dict(status=200, body=dict(pages=[], warnings=[], data=dict(transactions=[])))
    store = JobStore(settings, runner)
    first = store.submit([PAYLOAD], manifest(), CONTEXT, SCHEMA)
    second = store.submit([PAYLOAD], manifest(), CONTEXT, SCHEMA)
    assert first['jobs'][0]['run_id'] == second['jobs'][0]['run_id']
    assert store.result(first['submission_id'])[1] == 202
    gate.set()
    wait_ready(store, first['submission_id'])
    assert len(calls) == 1
    assert store.result(first['submission_id'])[1] == 200
    run_id = first['jobs'][0]['run_id']
    directory = store.runs[run_id]['directory']
    store.acknowledge(first['submission_id'])
    assert directory.exists()
    assert store.result(second['submission_id'])[1] == 200
    store.acknowledge(second['submission_id'])
    assert not directory.exists()
    assert store.acknowledge(second['submission_id']) == {'acknowledged': True}
    with pytest.raises(ServiceError):
        store.snapshot(first['submission_id'])


def test_context_changes_start_new_run_and_hash_mismatch_is_rejected(settings):
    def runner(*args):
        return dict(status=200, body={})
    store = JobStore(settings, runner)
    first = store.submit([PAYLOAD], manifest(), CONTEXT, SCHEMA)
    second = store.submit([PAYLOAD], manifest(), {**CONTEXT, 'default_currency': 'CAD'}, SCHEMA)
    assert first['jobs'][0]['job_id'] == second['jobs'][0]['job_id']
    assert first['jobs'][0]['run_id'] != second['jobs'][0]['run_id']
    with pytest.raises(ServiceError, match='job_id'):
        store.submit([b'changed'], manifest(), CONTEXT, SCHEMA)
    wait_ready(store, second['submission_id'])


def test_cleanup_cancels_old_runs_and_removes_unacknowledged_results(settings):
    finished = threading.Event()
    def runner(directory, cancelled):
        while not cancelled.wait(.01):
            pass
        finished.set()
        return dict(status=504, body=dict(error=dict(message='Expired')))
    store = JobStore(settings, runner)
    result = store.submit([PAYLOAD], manifest(), CONTEXT, SCHEMA)
    store.cleanup(time.time() + 86401)
    assert finished.wait(2) or not store.runs
    with pytest.raises(ServiceError):
        store.snapshot(result['submission_id'])


def test_http_background_job_real_worker_and_ack(transaction_ranker):
    settings, state = transaction_ranker
    app = create_app(settings)
    client = app.test_client()
    headers = {'Authorization': 'Bearer ' + settings.api_key}
    response = client.post('/v1/transaction-jobs', headers=headers, data=dict(
        documents=(BytesIO(PAYLOAD), 'rows.csv'), manifest=json.dumps(manifest()), context=json.dumps(CONTEXT), output_format=json.dumps(SCHEMA)))
    assert response.status_code == 202, response.json
    identity = response.json['submission_id']
    store = app.extensions['transaction_jobs']
    wait_ready(store, identity)
    result = client.get('/v1/transaction-jobs/' + identity + '/result', headers=headers)
    assert result.status_code == 200
    assert result.json['files'][0]['error'] is None, result.json
    row, = result.json['files'][0]['result']['data']['transactions']
    assert row['amount'] == '12.34'
    assert client.get('/v1/transaction-jobs/' + identity).status_code == 401
    assert client.post('/v1/transaction-jobs/' + identity + '/ack', headers=headers).status_code == 200
    assert client.post('/v1/transaction-jobs/' + identity + '/ack', headers=headers).status_code == 200
    assert client.get('/v1/transaction-jobs/' + identity, headers=headers).status_code == 404


def test_service_restart_does_not_reuse_jobs(settings):
    first = JobStore(settings, lambda *args: dict(status=200, body={}))
    result = first.submit([PAYLOAD], manifest(), CONTEXT, SCHEMA)
    wait_ready(first, result['submission_id'])
    second = JobStore(settings, lambda *args: None)
    assert first.instance != second.instance
    with pytest.raises(ServiceError):
        second.snapshot(result['submission_id'])
