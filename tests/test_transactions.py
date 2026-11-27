from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import threading
import time

from PIL import Image
import pytest

from ocr_processor.app import create_app
from ocr_processor.csv_documents import csv_rows
from ocr_processor.schema import validate_schema
from ocr_processor.transactions import amounts, dates, extract_transactions, validate_context
from ocr_processor.errors import ServiceError
from ocr_processor.worker import process


CONTEXT = {'accounts': [{'id': 'checking', 'name': 'Checking', 'description': 'Daily purchases',
                         'type': 'Checking', 'account_number': '1234', 'currency': 'USD'}],
           'categories': [{'id': 'food', 'name': 'Food', 'description': 'Groceries'}],
           'rules': [{'name': 'Markets', 'conditions': [{'field': 'description', 'operator': 'contains', 'value': 'Market'}],
                      'actions': {'category_id': 'food'}}],
           'currencies': [{'code': 'USD', 'precision': 2}]}
FIELDS = ['date', 'description', 'amount', 'currency', 'account_id', 'account_number',
          'source_account_id', 'destination_account_id', 'category_id', 'direction', 'is_refund',
          'document', 'page', 'review_notes']
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['transactions'], 'properties': {
    'transactions': {'type': 'array', 'items': {'type': 'object', 'required': FIELDS, 'additionalProperties': False,
        'properties': {name: {'type': 'boolean' if name == 'is_refund' else 'integer' if name in ('document', 'page') else ['string', 'null']} for name in FIELDS}}}}}


@pytest.fixture
def transaction_ranker(settings):
    state = {'requests': [], 'unknown': False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['requests'].append(request)
            choices = {'record_type': 'transaction', 'currency': 'USD', 'account_id': 'checking',
                       'source_account_id': 'checking', 'destination_account_id': '__unknown__',
                       'category_id': 'food', 'direction': 'outgoing', 'refund': 'no'}
            choices['account_id'] = state.get('accounts', {}).get(request['state']['pages'][0]['document'], 'checking')
            fields = request['state'].get('candidate', {}).get('fields', {})
            if fields.get('Description') == 'Payment':
                choices['direction'] = 'incoming'
            if fields.get('Description') == 'Opening balance':
                choices['record_type'] = 'exclude'
            for name in ('date', 'amount'):
                if name in request['questions']:
                    choices[name] = next((key for key in request['questions'][name]['criteria'] if key != '__unknown__'), '__unknown__')
            if state.get('amount'):
                choices['amount'] = state['amount']
            if state['unknown']:
                choices = {key: '__unknown__' for key in request['questions']}
            response = {'model': 'kev-test', 'answers': {key: {'type': 'choice', 'choice': choices[key]} for key in request['questions']}}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield replace(settings, llm_base_url='', llm_model='', systemone_ip='127.0.0.1', systemone_port=server.server_port), state
    server.shutdown()
    server.server_close()
    thread.join()


def upload(settings, content, filename='statement.pdf', **overrides):
    return create_app(settings).test_client().post('/v1/transactions',
        headers={'Authorization': 'Bearer ' + settings.api_key},
        data={'documents': (BytesIO(content), filename), 'context': json.dumps(CONTEXT),
              'output_format': json.dumps(SCHEMA), **overrides})


def test_pdf_actual_worker_uses_systemone_without_llm(transaction_ranker, pdf_bytes):
    settings, state = transaction_ranker
    response = upload(settings, pdf_bytes)
    assert response.status_code == 200, response.json
    row, = response.json['data']['transactions']
    assert row['amount'] == '12.34' and row['date'] == '2026-11-15'
    assert row['source_account_id'] == 'checking' and row['category_id'] == 'food'
    assert state['requests'][0]['state']['context'] == CONTEXT
    assert 'LLM' not in json.dumps(state['requests'])
    capabilities = create_app(settings).test_client().get('/v1/capabilities', headers={'Authorization': 'Bearer ' + settings.api_key}).json
    assert capabilities['transaction_extraction_configured'] and not capabilities['extraction_configured']


def test_csv_rows_mirrored_and_summary_preserved_until_decision(transaction_ranker):
    settings, state = transaction_ranker
    content = b'Date,Description,Amount\r\n2026-11-15,"Market, shop",-12.34\r\n2026-11-15,Payment,12.34\r\n2026-11-01,Opening balance,999.00\r\n'
    response = upload(settings, content, 'transactions.csv')
    assert response.status_code == 200, response.json
    rows = response.json['data']['transactions']
    assert len(rows) == 2 and len(state['requests']) == 4
    assert rows[0]['description'] == 'Market, shop'
    assert rows[0]['amount'] == rows[1]['amount'] == '12.34'
    assert rows[0]['source_account_id'] == 'checking' and rows[0]['destination_account_id'] is None
    assert rows[1]['source_account_id'] is None and rows[1]['destination_account_id'] == 'checking'
    assert sum('account_id' in request['questions'] for request in state['requests']) == 1
    assert all('currency' not in request['questions'] for request in state['requests'])
    assert all(not {'source_account_id', 'destination_account_id'} & request['questions'].keys() for request in state['requests'])
    assert response.json['pages'][0]['method'] == 'csv'


def test_receipt_image_total_is_one_transaction(transaction_ranker, tmp_path, monkeypatch):
    settings, state = transaction_ranker
    state['amount'] = '13.50'
    path = tmp_path / 'receipt.png'
    Image.new('RGB', (30, 30), 'white').save(path)
    monkeypatch.setattr('ocr_processor.documents.recognize', lambda *args: 'Market\n2026-11-15\nItem 10.00\nTax 3.50\nTOTAL USD 13.50')
    result = process({'mode': 'transactions', 'files': [str(path)], 'context': CONTEXT, 'schema': SCHEMA, 'force_ocr': False}, tmp_path, settings)
    row, = result['data']['transactions']
    assert row['amount'] == '13.50' and result['pages'][0]['method'] == 'ocr'


def test_unknown_values_and_schema_validation(transaction_ranker):
    settings, state = transaction_ranker
    state['unknown'] = True
    pages = [{'document': 1, 'page': 1, 'method': 'embedded', 'text': 'Unclear receipt'}]
    data, warnings = extract_transactions(pages, CONTEXT, validate_schema(SCHEMA), settings, time.monotonic() + 10)
    assert data['transactions'][0]['amount'] is None
    assert data['transactions'][0]['category_id'] is None
    assert data['transactions'][0]['currency'] == 'USD'
    assert data['transactions'][0]['account_id'] is None
    assert data['transactions'][0]['source_account_id'] is None
    assert data['transactions'][0]['destination_account_id'] is None
    assert warnings
    with pytest.raises(ServiceError, match='output_format'):
        extract_transactions(pages, CONTEXT, validate_schema({'type': 'object', 'required': ['missing']}), settings, time.monotonic() + 10)


@pytest.mark.parametrize('payload', [b'junk', b'a,b\n1,2,3\n', b'a,a\n1,2\n', b'a,b\n1,\x00\n'])
def test_reject_malformed_csv(payload):
    assert csv_rows(payload) is None


def test_csv_encoding_dates_and_integer_amounts():
    assert csv_rows('Date;Amount\n2026-11-15;12\n'.encode('utf-16'))[1][1] == '12'
    assert set(dates('11/15', 'Statement 2026')) == {'2026-11-15'}
    assert dates('11/15', 'No statement year') == {}
    assert set(dates('15 November', 'Statement 2026')) == {'2026-11-15'}
    assert set(dates('2026-11-15T12:30:00', '')) == {'2026-11-15'}
    assert set(dates('03/04/2026', '')) == {'2026-03-04', '2026-04-03'}
    assert set(amounts('', {'Amount': '-12'})) == {'12'}
    assert set(amounts('', {'Amount': '12,50'})) == {'12.50'}
    assert set(amounts('EUR 1.234,56')) == {'1234.56'}
    assert set(amounts('USD 1,234.56')) == {'1234.56'}
    assert set(amounts('JPY 1000')) == {'1000'}


@pytest.mark.parametrize('context', [[], {'accounts': [{'id': 'a'}, {'id': 'a'}]}, {'accounts': [None]}, {'categories': [{'id': str(i)} for i in range(255)]}, {'default_currency': None}, {'default_currency': 'usd'}])
def test_invalid_context_rejected(context):
    with pytest.raises(ServiceError):
        validate_context(context)


def test_transaction_endpoint_requires_systemone_and_schema(settings, transaction_ranker):
    assert upload(settings, b'a,b\n1,2\n').status_code == 503
    configured, _ = transaction_ranker
    assert upload(configured, b'a,b\n1,2\n', output_format='not-json').status_code == 400
    assert upload(configured, b'a,b\n1,2\n', instructions='unsupported').status_code == 400


def test_extended_timeouts_are_configurable(monkeypatch):
    from ocr_processor.config import Settings
    for key in ('REQUEST_TIMEOUT_SECONDS', 'SYSTEMONE_TIMEOUT_SECONDS', 'OCR_TIMEOUT_SECONDS'):
        monkeypatch.delenv(key, raising=False)
    settings = Settings.from_env()
    assert (settings.timeout, settings.systemone_timeout, settings.ocr_timeout) == (1800, 600, 120)
    monkeypatch.setenv('SYSTEMONE_TIMEOUT_SECONDS', '900')
    assert Settings.from_env().systemone_timeout == 900
    with pytest.raises(RuntimeError, match='timeouts'):
        create_app(replace(settings, api_key='test-secret-' * 4, systemone_timeout=0))


def test_systemone_timeout_uses_call_limit_and_remaining_batch_budget(transaction_ranker, monkeypatch):
    import httpx
    settings, _ = transaction_ranker
    client = httpx.Client
    timeouts = []

    def capture(**kwargs):
        timeouts.append(kwargs['timeout'])
        return client(**kwargs)

    monkeypatch.setattr('ocr_processor.systemone.httpx.Client', capture)
    pages = [{'document': 1, 'page': 1, 'method': 'embedded', 'text': '2026-11-15 Market USD 12.34'}]
    for remaining in (1800, 10):
        extract_transactions(pages, CONTEXT, validate_schema(SCHEMA), settings, time.monotonic() + remaining)
    assert timeouts[0] == 600
    assert timeouts[1] == 600
    assert all(0 < value <= 10 for value in timeouts[2:])


def test_account_selected_once_across_pages_and_separately_per_document(transaction_ranker):
    settings, state = transaction_ranker
    state['accounts'] = {1: 'checking', 2: 'savings'}
    context = {**CONTEXT, 'default_currency': 'CAD',
               'accounts': CONTEXT['accounts'] + [{'id': 'savings', 'name': 'Savings', 'currency': 'EUR'}]}
    pages = [{'document': document, 'page': page, 'method': 'embedded',
              'text': '2026-11-15 Market EUR 12.34'} for document, page in [(1, 1), (2, 1), (1, 2)]]
    data, _ = extract_transactions(pages, context, validate_schema(SCHEMA), settings, time.monotonic() + 10)
    rows = data['transactions']
    assert [row['account_id'] for row in rows] == ['checking', 'savings', 'checking']
    assert [row['source_account_id'] for row in rows] == ['checking', 'savings', 'checking']
    assert all(row['currency'] == 'CAD' and row['destination_account_id'] is None for row in rows)
    account_requests = [request for request in state['requests'] if 'account_id' in request['questions']]
    assert len(account_requests) == 2
    assert [page['page'] for page in account_requests[0]['state']['pages']] == [1, 2]
    assert all(set(request['questions']) == {'account_id'} for request in account_requests)
    assert all('currency' not in request['questions'] for request in state['requests'])
