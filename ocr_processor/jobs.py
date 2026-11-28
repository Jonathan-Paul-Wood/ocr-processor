"""Process-local, request-independent file jobs with acknowledged result delivery."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
from uuid import uuid4

from flask import jsonify, request

from .errors import ServiceError, unexpected_error
from .schema import load_json, validate_schema
from .transactions import validate_context

EXTRACTION_VERSION = 'transactions-batched-v1'


class JobStore:
    def __init__(self, settings, runner):
        self.settings, self.runner = settings, runner
        self.instance = str(uuid4())
        self.root = tempfile.TemporaryDirectory(prefix='ocr-jobs-')
        self.runs, self.submissions, self.acknowledged = {}, {}, {}
        self.condition = threading.Condition(threading.RLock())
        self.thread = None

    def cleanup(self, now=None):
        now = time.time() if now is None else now
        with self.condition:
            expired = [key for key, value in self.runs.items() if now - value['created'] > 86400]
            for key in expired:
                self._remove(key)
            for key in list(self.submissions):
                if now - self.submissions[key]['created'] > 86400:
                    del self.submissions[key]
            self.acknowledged = {key: value for key, value in self.acknowledged.items() if now - value < 86400}

    def _remove(self, identity):
        run = self.runs.pop(identity, None)
        if run:
            run['cancelled'].set()
            # A running subprocess owns its directory until it exits.
            if run['state'] != 'running':
                shutil.rmtree(run['directory'], ignore_errors=True)

    def submit(self, files, manifest, context, schema, force=False):
        fingerprint = hashlib.sha256(json.dumps(dict(context=context, schema=schema, force_ocr=force,
            model=self.settings.systemone_model, extraction_version=EXTRACTION_VERSION), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with self.condition:
            if len(self.runs) + len(files) > 500 or len(self.submissions) >= 500:
                raise ServiceError('busy', 'Job storage is full; retry after completed results are acknowledged.', 429)
            checked = []
            for payload, item in zip(files, manifest):
                identity = hashlib.sha256(payload).hexdigest()
                if not isinstance(item, dict) or item.get('job_id') != identity:
                    raise ServiceError('invalid_request', 'File content does not match its job_id.', 400)
                checked.append((payload, identity, item))
            current_bytes = sum(value['input_bytes'] for value in self.runs.values())
            needed = {identity: len(payload) for payload, identity, _ in checked
                      if not any(value['job_id'] == identity and value['fingerprint'] == fingerprint and value['state'] != 'failed'
                                 for value in self.runs.values())}
            if current_bytes + sum(needed.values()) > self.settings.max_upload_bytes * self.settings.max_files:
                raise ServiceError('busy', 'Temporary file storage is full; retry after completed jobs are acknowledged.', 429)
            submission = dict(id=str(uuid4()), created=time.time(), jobs=[])
            for payload, identity, item in checked:
                run = next((value for value in self.runs.values() if value['job_id'] == identity and value['fingerprint'] == fingerprint and value['state'] != 'failed'), None)
                if run is None:
                    run_id = str(uuid4())
                    directory = Path(self.root.name) / run_id
                    directory.mkdir()
                    path = directory / 'document-0'
                    path.write_bytes(payload)
                    job = dict(mode='transactions', files=[str(path)], context=context, schema=schema, force_ocr=force)
                    (directory / 'job.json').write_text(json.dumps(job), encoding='utf-8')
                    run = dict(run_id=run_id, job_id=identity, fingerprint=fingerprint, created=time.time(),
                               state='queued', directory=directory, cancelled=threading.Event(), result=None, error=None, input_bytes=len(payload))
                    self.runs[run_id] = run
                if not any(item['run_id'] == run['run_id'] for item in submission['jobs']):
                    submission['jobs'].append(dict(job_id=identity, run_id=run['run_id'], filename=str(item.get('filename', 'Document'))[:255]))
            self.submissions[submission['id']] = submission
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._work, daemon=True, name='ocr-file-jobs')
                self.thread.start()
            self.condition.notify_all()
            return self.snapshot(submission['id'])

    def _work(self):
        while True:
            with self.condition:
                run = next((value for value in self.runs.values() if value['state'] == 'queued'), None)
                if run is None:
                    self.condition.wait()
                    continue
                run['state'] = 'running'
            try:
                output = self.runner(run['directory'], run['cancelled'])
                with self.condition:
                    if output['status'] == 200:
                        run.update(state='succeeded', result=output['body'])
                    else:
                        run.update(state='failed', error=output['body'].get('error', {'message': 'File processing failed.'}))
            except Exception as exc:
                error = exc if isinstance(exc, ServiceError) else unexpected_error(exc, 'file_job')
                with self.condition:
                    run.update(state='failed', error=error.payload()['error'])
            finally:
                if run['cancelled'].is_set():
                    shutil.rmtree(run['directory'], ignore_errors=True)

    def snapshot(self, identity):
        with self.condition:
            submission = self.submissions.get(identity)
            if submission is None:
                raise ServiceError('job_unavailable', 'Import jobs expired, were acknowledged, or OCR Processor restarted. Upload again.', 404)
            jobs = []
            for item in submission['jobs']:
                run = self.runs.get(item['run_id'])
                if run is None:
                    raise ServiceError('job_unavailable', 'An import job expired. Upload again.', 404)
                progress = dict(status='processing_file', completed=0, total=None)
                try:
                    progress.update(json.loads((run['directory'] / 'progress.json').read_text(encoding='utf-8')))
                except (OSError, ValueError):
                    pass
                if run['state'] == 'succeeded':
                    progress['status'] = 'completed'
                jobs.append(dict(**item, **progress, state=run['state'], error=run['error']))
            return dict(submission_id=identity, service_instance_id=self.instance, jobs=jobs,
                        ready=all(item['state'] in ('succeeded', 'failed') for item in jobs),
                        completed_files=sum(item['state'] == 'succeeded' for item in jobs), total_files=len(jobs))

    def result(self, identity):
        with self.condition:
            snapshot = self.snapshot(identity)
            if not snapshot['ready']:
                return snapshot, 202
            files = []
            for item in snapshot['jobs']:
                run = self.runs[item['run_id']]
                files.append(dict(job_id=item['job_id'], run_id=item['run_id'], result=deepcopy(run['result']), error=run['error']))
            return dict(**snapshot, files=files), 200

    def acknowledge(self, identity):
        with self.condition:
            if identity in self.acknowledged:
                return {'acknowledged': True}
            snapshot = self.snapshot(identity)
            if not snapshot['ready']:
                raise ServiceError('job_unfinished', 'Cannot acknowledge unfinished jobs.', 409)
            submission = self.submissions.pop(identity)
            self.acknowledged[identity] = time.time()
            referenced = {job['run_id'] for value in self.submissions.values() for job in value['jobs']}
            for job in submission['jobs']:
                if job['run_id'] not in referenced:
                    self._remove(job['run_id'])
            return {'acknowledged': True}


def register_jobs(app, store, settings):
    @app.before_request
    def cleanup_jobs():
        if request.path.startswith('/v1/transaction-jobs'):
            store.cleanup()

    @app.post('/v1/transaction-jobs')
    def submit_jobs():
        if not settings.systemone_ip:
            raise ServiceError('systemone_unavailable', 'Configure SystemOne transaction extraction.', 503)
        if request.mimetype != 'multipart/form-data':
            raise ServiceError('invalid_request', 'Use multipart/form-data.', 400)
        if set(request.form) - {'manifest', 'context', 'output_format', 'force_ocr'} or set(request.files) - {'documents'}:
            raise ServiceError('invalid_request', 'Unknown request fields.', 400)
        if any(len(request.form.getlist(key)) != 1 for key in request.form):
            raise ServiceError('invalid_request', 'Duplicate form fields are not allowed.', 400)
        try:
            if len(request.form.get('context', '')) > 64000 or len(request.form.get('output_format', '')) > 32000:
                raise ValueError()
            manifest = load_json(request.form.get('manifest', ''))
            context = load_json(request.form.get('context', '{}'))
            schema = load_json(request.form.get('output_format', ''))
        except (ValueError, RecursionError):
            raise ServiceError('invalid_request', 'Supply bounded manifest, context and output_format JSON.', 400) from None
        validate_context(context)
        validate_schema(schema)
        uploads = request.files.getlist('documents')
        if not isinstance(manifest, list) or not 1 <= len(uploads) <= settings.max_files or len(manifest) != len(uploads):
            raise ServiceError('invalid_request', 'Supply matching manifest entries and documents.', 400)
        force = request.form.get('force_ocr', 'false')
        if force not in ('true', 'false'):
            raise ServiceError('invalid_request', 'force_ocr must be true or false.', 400)
        files = [upload.read() for upload in uploads]
        if any(not payload for payload in files):
            raise ServiceError('invalid_document', 'Empty documents are not supported.', 400)
        return jsonify(store.submit(files, manifest, context, schema, force == 'true')), 202

    @app.get('/v1/transaction-jobs/<identity>')
    def job_status(identity):
        return jsonify(store.snapshot(identity))

    @app.get('/v1/transaction-jobs/<identity>/result')
    def job_result(identity):
        body, status = store.result(identity)
        return jsonify(body), status

    @app.post('/v1/transaction-jobs/<identity>/ack')
    def acknowledge_jobs(identity):
        return jsonify(store.acknowledge(identity))
