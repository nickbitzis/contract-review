import json
from types import SimpleNamespace

import pytest

from contractreview.analyze import Pipeline
from contractreview.corpus import build_corpus
from contractreview.playbook import load_playbook

_PB = load_playbook()
_CORPUS = build_corpus(playbook=_PB)


@pytest.fixture
def playbook():
    return _PB


@pytest.fixture
def corpus():
    return _CORPUS


@pytest.fixture
def pipeline(playbook):
    return Pipeline(playbook)


class FakeClaude:
    """Stands in for anthropic.Anthropic(): returns a canned JSON body and records each request."""

    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.calls, self.stop_reason = payload, [], stop_reason
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        body = self.payload(kwargs) if callable(self.payload) else self.payload
        text = body if isinstance(body, str) else json.dumps(body)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=self.stop_reason,
                               usage=SimpleNamespace(input_tokens=900, output_tokens=120))


MSA = """MASTER SERVICES AGREEMENT

between Brightwater Mills ("Customer") and Tessavel, Inc. ("Supplier")

1. Definitions
1.1 "Fees" means the amounts payable under an Order Form.

2. Fees and Payment
2.1 Customer shall pay all undisputed invoices within sixty (60) days of receipt.
2.2 Late payments bear interest at 1.5% per month.

3. Limitation of Liability
3.1 In no event will either party's total liability under this Agreement exceed three times (3x) the annual fees payable under this Agreement.

4. Indemnification
Each party shall defend, indemnify, and hold harmless the other party from and against any third-party claim alleging that the Services infringe any patent, and shall pay all damages finally awarded.

5. Term and Termination
5.1 Either party may terminate this Agreement if the other party materially breaches it and does not cure the breach within thirty (30) days after notice.
5.2 Customer may terminate this Agreement for convenience on fifteen (15) days' prior written notice to Supplier.

6. Confidentiality
6.1 Each party shall protect the other party's Confidential Information. These obligations continue indefinitely.

7. Warranties
7.1 Supplier warrants that the Services will perform materially in accordance with the Documentation for ninety (90) days.

8. Governing Law
This Agreement is governed by the laws of the State of New York.
"""


@pytest.fixture
def msa_text():
    return MSA
