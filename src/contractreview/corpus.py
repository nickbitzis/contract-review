"""Seeded synthetic contract corpus with ground truth labels.

Every contract is written from clause templates in this file, for a made-up vendor (Tessavel) reviewing customer
paper from made-up companies. No real contract text is used. Each contract records, for every clause type it
contains, the true parameters and the character spans of the paragraphs that state them. The ground truth
deviations come from applying the playbook to the true parameters, so the eval measures how well the pipeline
finds and reads the clauses, not whether the playbook is right.

Messiness on purpose: six numbering styles, numbers that skip, clauses split across sections (liability cap and
its carve-outs, warranty and disclaimer, termination for convenience in its own section), terms defined in the
definitions section and used later ("the Liability Cap"), unusual headings ("Maximum Recovery", "Third-Party
Claims"), governing law buried in a miscellaneous section, ALL CAPS paragraphs, order forms in key: value form,
and distractor sections full of days, hours, and notice language.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .playbook import Playbook, load_playbook

SEED = 20261107
N_MSA, N_DPA, N_ORDER = 24, 8, 8

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORDINALS = {30: "thirtieth", 45: "forty-fifth", 60: "sixtieth", 90: "ninetieth"}

FIRST = ["Ashby", "Corrin", "Dunmore", "Fenwick", "Galloway", "Halden", "Ivers", "Kestrel", "Larkspur", "Morrow",
         "Northam", "Pellam", "Quarry", "Redfern", "Stillwater", "Thorne", "Umberly", "Vantry", "Wexley", "Yarrow"]
SECOND = ["Logistics", "Health Partners", "Foods", "Capital", "Energy", "Retail Group", "Media", "Manufacturing",
          "Insurance", "Hospitality"]
CITIES = {"Delaware": "Wilmington, Delaware", "New York": "New York County, New York",
          "California": "San Francisco County, California", "Texas": "Travis County, Texas",
          "Illinois": "Cook County, Illinois", "England and Wales": "London", "Ontario": "Toronto, Ontario"}


def words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (f"-{_ONES[o]}" if o else "")
    h, r = divmod(n, 100)
    return f"{_ONES[h]} hundred" + (f" {words(r)}" if r else "")


def wn(n: int) -> str:
    """'thirty (30)' style, the way contracts write numbers."""
    return f"{words(n)} ({n})"


def roman(n: int) -> str:
    out, vals = "", [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    for v, s in vals:
        while n >= v:
            out, n = out + s, n - v
    return out


@dataclass
class Para:
    text: str
    types: tuple[str, ...] = ()
    unusual: bool = False


@dataclass
class Section:
    heading: str
    paras: list[Para]
    key: str = ""  # used for cross references


@dataclass
class TruthClause:
    type: str
    params: dict
    spans: list[list[int]] = field(default_factory=list)
    phrasing: str = "standard"  # "unusual" when a paragraph was reworded by rephrase()


@dataclass
class Contract:
    id: str
    title: str
    kind: str  # msa | dpa | order_form
    customer: str
    text: str
    style: str
    clauses: dict[str, TruthClause]
    deviations: list[dict]
    missing: list[dict]

    def truth(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


class _Gen:
    def __init__(self, rng: random.Random, kind: str, customer: str):
        self.r, self.kind, self.customer = rng, kind, customer
        self.Cs = customer + ("'" if customer.endswith("s") else "'s")
        self.P = rng.choice(["Provider", "Supplier", "Vendor"])
        self.params: dict[str, dict] = {}
        self.defs: list[Para] = []

    def pick(self, *opts, weights=None):
        return self.r.choices(opts, weights=weights)[0]

    # ---- clause builders: each returns sections and records true params ----
    def lol(self) -> list[Section]:
        P = self.P
        uncapped = self.r.random() < 0.12
        if uncapped:
            text = self.pick(
                f"Nothing in this Agreement limits or excludes either party's liability for any claim arising out of "
                f"or related to this Agreement, and {P}'s liability under this Agreement shall be unlimited.",
                f"The parties agree that no limitation of liability applies to {P}'s obligations under this "
                f"Agreement, and {P}'s liability is uncapped.")
            self.params["limitation_of_liability"] = {"cap_multiple": None, "uncapped": True,
                                                      "data_breach_carveout": False}
            return [Section(self.pick("Limitation of Liability", "Liability"),
                            [Para(text, ("limitation_of_liability",))], "lol")]
        mult = self.pick(0.5, 1.0, 2.0, 3.0, 5.0, weights=[1, 4, 3, 2, 1])
        months = int(mult * 12)
        pct = {0.5: "fifty percent (50%) of", 1.0: "one hundred percent (100%) of", 2.0: "two times (2x)",
               3.0: "three times (3x)", 5.0: "five times (5x)"}[mult]
        variant = self.pick("months", "multiple", "exposure", "defined", weights=[4, 3, 2, 2])
        heading = "Limitation of Liability"
        lol = ("limitation_of_liability",)
        if variant == "months":
            text = (f"Except for the Excluded Claims, each party's aggregate liability arising out of or relating to "
                    f"this Agreement shall not exceed the fees paid or payable by {self.customer} in the "
                    f"{wn(months)} months preceding the event giving rise to the claim.")
        elif variant == "multiple":
            text = (f"In no event will either party's total liability under this Agreement exceed {pct} the annual "
                    f"fees payable under this Agreement.")
        elif variant == "exposure":
            heading = "Maximum Recovery"
            text = (f"Neither party's aggregate exposure for claims connected with this Agreement will be greater "
                    f"than the amounts paid by {self.customer} during the {wn(months)} month period immediately "
                    f"before the claim.")
        else:
            self.defs.append(Para(f'"Liability Cap" means an amount equal to the fees paid or payable by '
                                  f"{self.customer} in the {wn(months)} months before the claim.", lol))
            text = "Each party's total liability arising out of this Agreement is limited to the Liability Cap."
        paras = [Para(text, lol)]
        paras.append(Para("Neither party is liable for any indirect, incidental, special, or consequential damages, "
                          "including lost profits, even if advised of their possibility.", lol))
        carve = self.r.random() < 0.75
        breach = carve and self.r.random() < 0.45
        self.params["limitation_of_liability"] = {"cap_multiple": mult, "uncapped": False,
                                                  "data_breach_carveout": breach}
        out = [Section(heading, paras, "lol")]
        if carve:
            ctext = ("The limitations in Section [[ref:lol]] do not apply to (a) a party's indemnification "
                     "obligations, (b) breach of confidentiality obligations, or (c) gross negligence or willful "
                     "misconduct")
            ctext += f", or (d) {P}'s breach of its data protection or security obligations." if breach else "."
            if self.r.random() < 0.5:
                out.append(Section(self.pick("Exclusions", "Excluded Claims"), [Para(ctext, lol)], "lol_x"))
            else:
                paras.append(Para(ctext.replace("Section [[ref:lol]]", "this Section"), lol))
        return out

    def indemnity(self) -> list[Section]:
        P, C = self.P, self.customer
        mutual = self.r.random() < 0.55
        scope = self.pick("ip_only", "ip_and_data", "any_claim", weights=[5, 2, 2])
        who = ('Each party (the "Indemnifying Party") shall defend, indemnify, and hold harmless the other party'
               if mutual else f"{P} shall defend, indemnify, and hold harmless {C} and its affiliates, officers, "
                              f"and employees")
        if self.r.random() < 0.25:
            who = who.replace("defend, indemnify, and hold harmless", "defend and hold harmless")
        what = {"ip_only": "from and against any third-party claim alleging that the Services infringe or "
                           "misappropriate any patent, copyright, trademark, or trade secret",
                "ip_and_data": "from and against any third-party claim alleging that the Services infringe or "
                               "misappropriate any intellectual property right, or arising from a breach of the "
                               "data protection obligations in this Agreement",
                "any_claim": "from and against any and all claims, losses, liabilities, and expenses arising out of "
                             "or relating to this Agreement or the Services"}[scope]
        self.params["indemnification"] = {"mutual": mutual, "scope": scope}
        ind = ("indemnification",)
        paras = [Para(f"{who} {what}, and shall pay all damages and costs finally awarded.", ind),
                 Para("The indemnified party shall give prompt written notice of the claim, sole control of the "
                      "defense, and reasonable cooperation at the indemnifying party's expense.", ind)]
        return [Section(self.pick("Indemnification", "Indemnity", "Third-Party Claims", weights=[5, 2, 2]), paras)]

    def termination(self, renewal_inside: list[Para] | None) -> list[Section]:
        P, C = self.P, self.customer
        cure = self.pick(10, 15, 30, 45, weights=[1, 2, 5, 1])
        tfc = self.pick(None, 15, 30, 60, 90, 120, weights=[5, 1, 2, 1, 2, 1])
        self.params["termination"] = {"tfc_notice_days": tfc, "cure_days": cure}
        t = ("termination",)
        paras = [Para("This Agreement begins on the Effective Date and continues until all Order Forms under it have "
                      "expired or been terminated.", t)]
        paras += renewal_inside or []
        paras.append(Para(self.pick(
            f"Either party may terminate this Agreement upon written notice if the other party materially breaches it "
            f"and does not cure the breach within {wn(cure)} days after receiving notice of it.",
            f"Either party may terminate this Agreement if the other party commits a material breach and fails to "
            f"remedy such breach within {cure} days of written notice."), t))
        tsec = []
        if tfc is None:
            if self.r.random() < 0.5:
                paras.append(Para("Neither party may terminate this Agreement for convenience.", t))
        else:
            tp = Para(self.pick(
                f"{C} may terminate this Agreement or any Order Form for convenience on {wn(tfc)} days' prior written "
                f"notice to {P}.",
                f"{C} may terminate this Agreement at any time, for any reason or no reason, by giving {P} at least "
                f"{tfc} days written notice.",
                f"{C} may end this Agreement without cause on {wn(tfc)} days' notice."), t)
            if self.r.random() < 0.3:
                tsec = [Section("Termination for Convenience", [tp])]
            else:
                paras.append(tp)
        paras.append(Para(f"Upon termination, {C} shall pay all fees accrued through the effective date of "
                          "termination, and each party shall return the other party's Confidential Information.", t))
        return [Section(self.pick("Term and Termination", "Termination", "Duration"), paras)] + tsec

    def renewal_paras(self) -> list[Para]:
        auto = self.r.random() < 0.8
        a = ("auto_renewal",)
        if not auto:
            self.params["auto_renewal"] = {"auto_renews": False, "renewal_notice_days": None}
            return [Para("The Subscription Term will not renew automatically. Any renewal requires a new Order Form "
                         "signed by both parties.", a)]
        n = self.pick(10, 15, 30, 45, 60, 90, weights=[1, 1, 3, 1, 3, 1])
        self.params["auto_renewal"] = {"auto_renews": True, "renewal_notice_days": n}
        return [Para(self.pick(
            f"This Agreement automatically renews for successive one (1) year periods unless either party gives "
            f"written notice of non-renewal at least {wn(n)} days before the end of the then-current term.",
            f"Each Subscription Term will renew for an additional twelve (12) months unless a party notifies the other "
            f"of its intent not to renew no later than {n} days prior to expiration."), a)]

    def payment(self, include_days: bool = True) -> list[Section]:
        P, C = self.P, self.customer
        days = self.pick(30, 45, 60, 90, weights=[3, 2, 3, 2])
        late = self.pick(None, 0.0, 0.5, 1.0, 1.5, weights=[3, 2, 1, 2, 2])
        self.params["payment_terms"] = {"payment_days": days, "late_fee_pct": late}
        p = ("payment_terms",)
        paras = [Para(f"{P} will invoice {C} annually in advance for the fees in each Order Form.", p)]
        if include_days:
            paras.append(Para(self.pick(
                f"{C} shall pay all undisputed invoices within {wn(days)} days of receipt.",
                f"Payment terms are net {days} days from the invoice date.",
                f"Invoices are due and payable {days} days after {C} receives them.",
                f"{C} will remit payment no later than the {ORDINALS[days]} day following the invoice date.",
                weights=[4, 3, 2, 2]), p))
        if late is not None:
            paras.append(Para(self.pick(
                f"Late payments bear interest at {late:g}% per month.",
                f"Overdue amounts accrue interest at the lesser of {late:g}% per month or the highest rate permitted "
                f"by law.") if late else "No late fees, interest, or penalties shall apply to any payment.", p))
        paras.append(Para("Fees exclude taxes. Each party is responsible for the taxes imposed on it by law.", p))
        return [Section(self.pick("Fees and Payment", "Payment", "Invoicing and Payment"), paras)]

    def governing(self) -> tuple[list[Section], Para]:
        law = self.pick("Delaware", "New York", "California", "Texas", "Illinois", "England and Wales", "Ontario",
                        weights=[5, 3, 2, 2, 1, 2, 1])
        self.params["governing_law"] = {"law": law}
        g = ("governing_law",)
        if law == "England and Wales":
            text = "This Agreement shall be construed in accordance with the laws of England and Wales, and the courts of London have exclusive jurisdiction."
        elif law == "Ontario":
            text = "This Agreement is governed by the laws of the Province of Ontario and the federal laws of Canada applicable there."
        else:
            text = (f"This Agreement is governed by the laws of the State of {law}, without regard to its conflict of "
                    f"laws principles. The state and federal courts located in {CITIES[law]} have exclusive "
                    f"jurisdiction.")
        para = Para(text, g)
        return [Section(self.pick("Governing Law", "Disputes", "Choice of Law"), [para])], para

    def confidentiality(self) -> list[Section]:
        n = self.pick(2, 3, 5, 7, 99, weights=[1, 4, 3, 1, 2])
        self.params["confidentiality"] = {"term_years": n}
        c = ("confidentiality",)
        if n == 99:
            tail = self.pick("These obligations survive termination of this Agreement in perpetuity.",
                             "These obligations continue indefinitely.")
        else:
            tail = self.pick(f"The obligations in this Section continue for {wn(n)} years after termination of this "
                             f"Agreement.", f"These obligations apply for a period of {n} years from the date of "
                                            f"disclosure.")
        paras = [Para("Each party may disclose Confidential Information to the other. The receiving party shall use "
                      "it only to perform this Agreement and shall protect it with at least reasonable care.", c),
                 Para(tail, c)]
        return [Section(self.pick("Confidentiality", "Non-Disclosure", "Confidential Information",
                                  weights=[4, 1, 2]), paras)]

    def data_protection(self, split: bool) -> list[Section]:
        P, C, Cs = self.P, self.customer, self.Cs
        hours = self.pick(None, 12, 24, 48, 72, weights=[1, 1, 3, 2, 3])
        sub = self.pick("notice", "objection", "consent", weights=[3, 3, 3])
        audit = self.pick("report", "annual", "onsite", weights=[3, 3, 2])
        self.params["data_protection"] = {"breach_hours": hours, "subprocessors": sub, "audit": audit}
        d = ("data_protection",)
        if hours is None:
            b = f"{P} shall notify {C} of any Security Incident without undue delay after becoming aware of it."
        elif hours in (24, 48) and self.r.random() < 0.35:
            days = hours // 24
            b = (f"{P} shall notify {C} of any Security Incident within {wn(days)} business "
                 f"day{'s' if days > 1 else ''} of becoming aware of it.")
        else:
            b = self.pick(f"{P} shall notify {C} of any Security Incident within {wn(hours)} hours of becoming aware "
                          f"of it.",
                          f"{P} shall report a Security Incident to {C} without undue delay, and in any event no later "
                          f"than {hours} hours after discovery.")
        s = {"notice": f"{C} authorizes {P} to engage subprocessors. {P} will maintain a list of subprocessors and "
                       f"notify {C} of any additions.",
             "objection": f"{P} will notify {C} at least thirty (30) days before engaging a new subprocessor, and {C} "
                          f"may object on reasonable data protection grounds.",
             "consent": self.pick(f"{P} shall not engage any subprocessor without {Cs} prior written consent.",
                                  f"No third party may process {C} Personal Data on {P}'s behalf unless {C} has "
                                  f"first approved that party in writing.")}[sub]
        a = {"report": f"{P} will provide its most recent SOC 2 Type II report on request, no more than once per "
                       f"year.",
             "annual": f"{C} may audit {P}'s compliance once per year on thirty (30) days' prior notice, by remote "
                       f"document review.",
             "onsite": f"{C} or its auditors may inspect {P}'s facilities, systems, and records on-site as often as "
                       f"{C} deems necessary."}[audit]
        if split:
            return [Section(self.pick("Security Incidents", "Breach Notification"), [Para(b, d)]),
                    Section(self.pick("Subprocessors", "Sub-processing"), [Para(s, d)]),
                    Section(self.pick("Audits", "Audit Rights", "Compliance Verification"), [Para(a, d)])]
        return [Section("Data Protection", [Para(f"{P} shall process {C} Personal Data only on {Cs} documented "
                                                  "instructions.", d), Para(b, d), Para(s, d), Para(a, d)])]

    def warranties(self) -> list[Section]:
        P, _C, Cs = self.P, self.customer, self.Cs
        days = self.pick(90, 180, 365, weights=[5, 3, 2])
        disclaimer = self.r.random() < 0.7
        self.params["warranties"] = {"warranty_days": days, "disclaimer": disclaimer}
        w = ("warranties",)
        period = {90: self.pick("ninety (90) days", "three (3) months"),
                  180: self.pick("one hundred eighty (180) days", "six (6) months"),
                  365: self.pick("one (1) year", "twelve (12) months")}[days]
        paras = [Para(f"{P} warrants that the Services will perform materially in accordance with the Documentation "
                      f"for {period} after the start of the Subscription Term.", w)]
        out = []
        if disclaimer:
            dp = Para(f"EXCEPT AS EXPRESSLY PROVIDED IN THIS AGREEMENT, {P.upper()} MAKES NO OTHER WARRANTIES AND "
                      "DISCLAIMS ALL IMPLIED WARRANTIES, INCLUDING ANY WARRANTY OF MERCHANTABILITY OR FITNESS FOR A "
                      "PARTICULAR PURPOSE.", w)
            if self.r.random() < 0.4:
                out = [Section("Disclaimer", [dp])]
            else:
                paras.append(dp)
        else:
            paras.append(Para(f"{P} further warrants that the Services are fit for {Cs} intended purpose.", w))
        return [Section(self.pick("Warranties", "Representations and Warranties", "Performance Warranty"),
                        paras)] + out

    def assignment(self) -> list[Section]:
        P, C, Cs = self.P, self.customer, self.Cs
        v = self.pick("mutual_coc", "mutual_consent", "one_sided", weights=[3, 3, 2])
        self.params["assignment"] = {"assignment": v}
        text = {"mutual_coc": "Neither party may assign this Agreement without the other party's prior written "
                              "consent, except to a successor in a merger, acquisition, or sale of all or "
                              "substantially all of its assets.",
                "mutual_consent": "Neither party may assign or transfer this Agreement without the prior written "
                                  "consent of the other party.",
                "one_sided": f"{P} may not assign this Agreement without {Cs} prior written consent. {C} may "
                             f"assign this Agreement freely."}[v]
        return [Section(self.pick("Assignment", "Transfer"), [Para(text, ("assignment",))])]

    def non_solicit(self) -> list[Section]:
        P, C = self.P, self.customer
        n = self.pick(6, 12, 18, 24, 36, weights=[1, 4, 2, 2, 1])
        self.params["non_solicit"] = {"nonsolicit_months": n}
        text = self.pick(
            f"During the term of this Agreement and for {wn(n)} months thereafter, neither party shall solicit for "
            f"employment any employee of the other party who worked on the Services.",
            f"{P} shall not hire or attempt to hire any {C} employee for {n} months after termination of this "
            f"Agreement.")
        return [Section(self.pick("Non-Solicitation", "Employees", "No Hire"), [Para(text, ("non_solicit",))])]

    def insurance(self) -> list[Section]:
        n = self.pick(1, 2, 5, 10, 20, weights=[1, 3, 3, 2, 1])
        self.params["insurance"] = {"coverage_millions": n}
        amount = self.pick(f"${n},000,000", f"${n} million", f"USD {n * 1_000_000:,}")
        text = (f"{self.P} shall maintain, at its own expense, commercial general liability and cyber liability "
                f"insurance with limits of not less than {amount} per occurrence.")
        return [Section("Insurance", [Para(text, ("insurance",))])]

    # ---- distractors ----
    def definitions(self) -> Section:
        C = self.customer
        base = [Para(f'"Agreement" means this agreement between {self.P} and {C}, including all Order Forms.'),
                Para('"Fees" means the amounts payable under an Order Form.'),
                Para('"Security Incident" means unauthorized access to Customer Personal Data in the Services.'),
                Para('"Subscription Term" means the period stated in an Order Form.')]
        return Section("Definitions", base + self.defs)

    def distractors(self) -> list[Section]:
        P, C, Cs = self.P, self.customer, self.Cs
        pool = [
            Section("Services", [Para(f"{P} will make the Services available to {C} during the Subscription Term in "
                                      "accordance with the Documentation.")]),
            Section("Support and Service Levels", [Para(f"{P} will respond to Severity 1 issues within four (4) hours "
                                                        "and to Severity 2 issues within one (1) business day.")]),
            Section("Notices", [Para("Notices must be in writing and are deemed received two (2) days after mailing "
                                     "or on delivery by email.")]),
            Section("Force Majeure", [Para("Neither party is liable for delays caused by events beyond its reasonable "
                                           "control. If such an event lasts more than thirty (30) days, either party "
                                           "may end the affected Order Form.")]),
            Section("Publicity", [Para(f"{P} may name {C} as a customer with {Cs} prior written approval.")]),
            Section("Export Compliance", [Para("Each party shall comply with applicable export control and sanctions "
                                               "laws.")]),
        ]
        self.r.shuffle(pool)
        return pool[: self.r.randint(3, 6)]


UNUSUAL_RATE = 0.3


def rephrase(ctype: str, text: str, params: dict, g: _Gen) -> str | None:
    """Reword a standard paragraph in plainer, less contract-like language with the same meaning.

    These phrasings were written after the parsers and the parsers were not tuned on them. They stand in for
    the long tail of customer drafting that a keyword and regex baseline does not anticipate.
    """
    P, C, Cs, pick = g.P, g.customer, g.Cs, g.pick
    Ps = P + "'s"
    low = text.lower()
    if ctype == "limitation_of_liability":
        m = params.get("cap_multiple")
        if m and re.search(r"exceed|greater than|limited to the liability cap", low) and "liability cap\"" not in low:
            if m == 0.5:
                return "The most either party can recover from the other, for all claims combined, is half of the annual subscription charges."
            yrs = int(m)
            return pick(f"The maximum amount either party may recover from the other for all claims in the aggregate is "
                        f"capped at the total charges paid over the prior {words(yrs)} contract year{'s' if yrs > 1 else ''}.",
                        f"Liability of each party is restricted to {int(m * 100)}% of the Fees for the then-current year.")
        return None
    if ctype == "indemnification" and "hold harmless" in low:
        scope, mutual = params["scope"], params["mutual"]
        what = {"ip_only": "asserting that the Services violate a third party's patent, copyright, or trade secret",
                "ip_and_data": "asserting that the Services violate a third party's patent or copyright, or that "
                               "personal data was mishandled",
                "any_claim": "connected in any way with the Services"}[scope]
        who = "The parties will each protect the other against" if mutual else f"{P} will cover {Cs} losses from"
        return f"{who} actions brought by outsiders {what}."
    if ctype == "termination":
        if "convenience" in low or "without cause" in low or "any reason" in low:
            n = params.get("tfc_notice_days")
            if n:
                return f"{C} may cancel this Agreement whenever it chooses by notifying {P} in writing {n} days in advance."
            return None
        if "cure" in low or "remedy" in low:
            return (f"If one party seriously fails to perform and has not corrected the failure within "
                    f"{params['cure_days']} days after being told about it, the other party may end this Agreement.")
        return None
    if ctype == "payment_terms":
        if re.search(r"undisputed invoices|net \d+|due and payable|remit payment", low):
            n = params["payment_days"]
            return pick(f"Amounts invoiced fall due {words(n)} days from the invoice date.",
                        f"{C} has {n} calendar days to settle each invoice.")
        if "interest" in low or "late fees" in low:
            p = params.get("late_fee_pct")
            if p:
                return f"Unpaid balances carry a monthly finance charge of {p:g} percent."
            return f"{P} will not charge anything extra on overdue invoices."
        return None
    if ctype == "auto_renewal":
        if params.get("auto_renews"):
            n = params["renewal_notice_days"]
            return f"Unless cancelled in writing {n} days ahead of the anniversary date, the subscription rolls over for another year."
        return "The subscription ends at the close of the Subscription Term unless the parties sign a renewal."
    if ctype == "governing_law":
        law = params["law"]
        return pick(f"Any dispute will be resolved under {law} law, and the parties consent to venue in {CITIES[law]}.",
                    f"The substantive law of {law} applies to this Agreement.")
    if ctype == "confidentiality" and re.search(r"years|perpetuity|indefinitely", low):
        n = params["term_years"]
        return "The duty of secrecy has no end date." if n == 99 else f"The duty of secrecy lasts {words(n)} years beyond expiry."
    if ctype == "data_protection":
        if "security incident" in low and params.get("breach_hours"):
            h = params["breach_hours"]
            return pick(f"{P} will tell {C} about any confirmed data breach within {h} hours.",
                        f"Notification of a Security Incident is due no later than {words(h)} hours after {P} learns of it.")
        if "subprocessor" in low or "third party may process" in low:
            return {"consent": f"{P} needs {Cs} written sign-off before any new subprocessor touches {C} data.",
                    "objection": f"{C} has the right to veto a new subprocessor during a thirty (30) day review window.",
                    "notice": f"{P} posts changes to its subprocessor list on its website."}[params["subprocessors"]]
        if re.search(r"audit|inspect|soc 2", low):
            return {"onsite": f"{C} may send its own staff to examine {Ps} data centers without limit.",
                    "annual": f"{C} may review {Ps} compliance records one time every twelve months.",
                    "report": f"In place of audits, {P} will share third-party attestation reports, such as ISO 27001 "
                              f"certificates."}[params["audit"]]
        return None
    if ctype == "warranties":
        if "disclaims all implied" in low:
            return "THE SERVICES ARE PROVIDED WITHOUT ANY OTHER EXPRESS OR IMPLIED GUARANTEES, INCLUDING SUITABILITY FOR A SPECIFIC USE."
        if "perform materially" in low:
            d = params["warranty_days"]
            period = {90: "three months", 180: "six months", 365: "a full year"}[d]
            return f"{P} stands behind the Services for {period}: if they do not work as documented, {P} will fix them."
        return None
    if ctype == "assignment":
        return {"mutual_coc": "No party may transfer its rights or duties under this Agreement without written approval, "
                              "though either party may do so to an entity that acquires its business.",
                "mutual_consent": "Rights under this Agreement may be transferred by a party only with the other's "
                                  "written approval.",
                "one_sided": f"{C} may transfer this Agreement to any affiliate or successor. {P} may not transfer it "
                             f"without {Cs} approval."}[params["assignment"]]
    if ctype == "non_solicit":
        n = params["nonsolicit_months"]
        return f"Neither party will recruit the other's personnel for {words(n)} months after the last Order Form ends."
    if ctype == "insurance":
        return f"{P} will carry cyber coverage of at least {params['coverage_millions']} million dollars per incident."
    return None


STYLES = ["decimal", "section", "article", "inline", "caps", "paren"]


def _render(title: str, header: list[str], sections: list[Section], style: str, rng: random.Random):
    """Render sections with one numbering style. Returns text and, per para object id, its (start, end)."""
    numbers, n = {}, 0
    for s in sections:
        n += rng.choice([1, 1, 1, 2]) if style == "decimal" else 1  # decimal numbering sometimes skips
        numbers[id(s)] = n
    refs = {s.key: numbers[id(s)] for s in sections if s.key}
    out: list[str] = [title, ""] + header + ([""] if header else [])
    pos = sum(len(x) + 1 for x in out)
    spans: dict[int, tuple[int, int]] = {}

    def emit(prefix: str, para: Para | None, suffix_text: str | None = None):
        nonlocal pos
        text = suffix_text if para is None else para.text
        for key, num in refs.items():
            text = text.replace(f"[[ref:{key}]]", str(num) if style != "article" else f"{num}.1")
        line = prefix + text
        if para is not None:
            spans[id(para)] = (pos + len(prefix), pos + len(line))
        out.append(line)
        pos += len(line) + 1

    for s in sections:
        num = numbers[id(s)]
        multi = len(s.paras) > 1
        if style == "inline":
            emit(f"{num}. {s.heading}. ", s.paras[0])
            for k, p in enumerate(s.paras[1:], 2):
                emit(f"{num}.{k}. ", p)
        else:
            head = {"decimal": f"{num}. {s.heading}", "section": f"Section {num}. {s.heading}",
                    "article": f"ARTICLE {roman(num)} - {s.heading.upper()}", "caps": s.heading.upper(),
                    "paren": f"{num}) {s.heading}"}[style]
            emit("", None, head)
            for k, p in enumerate(s.paras, 1):
                prefix = {"decimal": f"{num}.{k} " if multi else "", "section": f"({chr(96 + k)}) " if multi else "",
                          "article": f"{num}.{k} ", "caps": "", "paren": f"{num}.{k}) " if multi else ""}[style]
                emit(prefix, p)
        emit("", None, "")
    return "\n".join(out).rstrip() + "\n", spans


def _truth(g: _Gen, sections: list[Section], header_paras: list[Para], spans: dict, playbook: Playbook):
    clauses: dict[str, TruthClause] = {}
    paras = header_paras + [p for s in sections for p in s.paras]
    for t, params in g.params.items():
        mine = [p for p in paras if t in p.types and id(p) in spans]
        clauses[t] = TruthClause(t, params, [list(spans[id(p)]) for p in mine],
                                 "unusual" if any(p.unusual for p in mine) else "standard")
    deviations, missing = [], []
    for t, c in clauses.items():
        for chk in playbook.rules[t].checks:
            sev, band = chk.evaluate(c.params.get(chk.param))
            if sev != "ok":
                deviations.append({"type": t, "param": chk.param, "severity": sev, "band": band})
    for t in playbook.required(g.kind):
        if t not in clauses:
            missing.append({"type": t, "severity": playbook.rules[t].missing_severity})
    return clauses, deviations, missing


def _msa(g: _Gen, idx: int) -> tuple[list[Section], list[Para]]:
    r = g.r
    renewal_inside = r.random() < 0.5
    renewal = g.renewal_paras() if r.random() < 0.85 else []
    body: list[Section] = []
    body += g.payment()
    body += g.confidentiality()
    body += g.warranties()
    # A few contracts drop a required clause so missing-clause detection has something to find.
    if idx % 8 != 3:
        body += g.lol()
    body += g.indemnity()
    body += g.termination(renewal if renewal_inside else None)
    if renewal and not renewal_inside:
        body.append(Section(r.choice(["Renewal", "Automatic Renewal"]), renewal))
    if r.random() < 0.6:
        body += g.data_protection(split=False)
    if r.random() < 0.8:
        body += g.assignment()
    if r.random() < 0.5:
        body += g.non_solicit()
    if r.random() < 0.5:
        body += g.insurance()
    misc_paras = [Para("This Agreement is the entire agreement of the parties on its subject matter."),
                  Para("The parties are independent contractors.")]
    if idx % 11 != 5:
        gov, gpara = g.governing()
        if r.random() < 0.35:
            misc_paras.insert(1, gpara)  # governing law buried in Miscellaneous
        else:
            body += gov
    body += g.distractors()
    r.shuffle(body)  # customer paper puts things in its own order
    return [g.definitions()] + body + [Section("Miscellaneous", misc_paras)], []


def _dpa(g: _Gen) -> tuple[list[Section], list[Para]]:
    C, P, Cs = g.customer, g.P, g.Cs
    sections = [Section("Scope and Roles", [Para(f"{C} is the controller and {P} is the processor of {C} Personal "
                                                 "Data processed under the Agreement.")]),
                Section("Processing Instructions", [Para(f"{P} shall process {C} Personal Data only on {Cs} "
                                                         "documented instructions.")])]
    sections += g.data_protection(split=True)
    sections.append(Section("Return and Deletion", [Para(f"Within thirty (30) days after termination, {P} shall "
                                                         f"delete or return all {C} Personal Data.")]))
    if g.r.random() < 0.5:
        sections += g.confidentiality()
    if g.r.random() < 0.5:
        sections.append(Section("Liability", [Para("Each party's liability under this DPA is subject to the "
                                                   "limitations set forth in the Agreement.")]))
    if g.r.random() < 0.6:
        sections += g.governing()[0]
    g.r.shuffle(sections)
    return [g.definitions()] + sections, []


def _order(g: _Gen, idx: int) -> tuple[list[Section], list[Para]]:
    r = g.r
    days = r.choice([30, 45, 60, 90])
    header = [Para(f"Customer: {g.customer}"), Para(f"Order Number: OF-2026-{100 + idx}"),
              Para("Subscription Term: twelve (12) months from the Order Start Date")]
    if idx % 7 != 2:
        header.append(Para(f"Payment Terms: Net {days}", ("payment_terms",)))
    sections = g.payment(include_days=False)
    if idx % 7 != 2:
        g.params["payment_terms"]["payment_days"] = days
    else:
        del g.params["payment_terms"]
        sections = []
    ren = g.renewal_paras()
    rp = g.params["auto_renewal"]
    if r.random() < 0.5:
        line = ("Renewal: Automatic, " + f"{rp['renewal_notice_days']} days notice of non-renewal"
                if rp["auto_renews"] else "Renewal: No automatic renewal")
        header.append(Para(line, ("auto_renewal",)))
    else:
        sections.append(Section("Renewal", ren))
    header.append(Para("Governing Agreement: Tessavel Master Subscription Agreement"))
    if r.random() < 0.5:
        sections += g.insurance()
    if r.random() < 0.4:
        sections += g.non_solicit()
    return sections, header


def build_corpus(seed: int = SEED, playbook: Playbook | None = None) -> list[Contract]:
    playbook = playbook or load_playbook()
    rng = random.Random(seed)
    names = [f"{a} {b}" for a in FIRST for b in SECOND]
    rng.shuffle(names)
    out = []
    plan = [("msa", i) for i in range(N_MSA)] + [("dpa", i) for i in range(N_DPA)] + \
           [("order_form", i) for i in range(N_ORDER)]
    for n, (kind, idx) in enumerate(plan):
        r = random.Random(seed * 1000 + n)
        customer = names[n]
        g = _Gen(r, kind, customer)
        if kind == "msa":
            sections, header_paras = _msa(g, idx)
            title = r.choice(["MASTER SERVICES AGREEMENT", "Master Subscription Agreement", "SOFTWARE AS A SERVICE AGREEMENT"])
        elif kind == "dpa":
            sections, header_paras = _dpa(g)
            title = r.choice(["DATA PROCESSING AGREEMENT", "Data Processing Addendum"])
        else:
            sections, header_paras = _order(g, idx)
            title = "ORDER FORM"
        for sec in sections:
            for para in sec.paras:
                if para.types and r.random() < UNUSUAL_RATE:
                    new = rephrase(para.types[0], para.text, g.params.get(para.types[0], {}), g)
                    if new and new != para.text:
                        para.text, para.unusual = new, True
        style = STYLES[n % len(STYLES)] if kind != "order_form" else "decimal"
        preamble = [f"between {customer} (\"Customer\") and Tessavel, Inc. (\"{g.P}\")"]
        header_lines = preamble + [p.text for p in header_paras]
        text, spans = _render(title, header_lines, sections, style, r)
        # header paragraph spans: locate each header line in the rendered text
        for p in header_paras:
            start = text.index(p.text)
            spans[id(p)] = (start, start + len(p.text))
        clauses, deviations, missing = _truth(g, sections, header_paras, spans, playbook)
        cid = f"{kind.replace('_', '-')}-{idx + 1:02d}"
        out.append(Contract(cid, f"{title.title()} with {customer}", kind, customer, text, style, clauses,
                            deviations, missing))
    return out


def write_corpus(out_dir: Path, seed: int = SEED) -> list[Path]:
    out_dir = Path(out_dir)
    (out_dir / "contracts").mkdir(parents=True, exist_ok=True)
    paths = []
    corpus = build_corpus(seed)
    for c in corpus:
        p = out_dir / "contracts" / f"{c.id}.txt"
        p.write_text(c.text)
        paths.append(p)
    truth = out_dir / "truth.jsonl"
    truth.write_text("".join(json.dumps(c.truth()) + "\n" for c in corpus))
    return paths + [truth]
