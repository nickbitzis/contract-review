from contractreview.segment import detect_kind, segment, sentences


def heads(seg):
    return [(c.number, c.heading) for c in seg.clauses]


def test_decimal_with_skipped_numbers():
    text = "AGREEMENT\n\n1. Payment\n1.1 Pay in 30 days.\n1.2 Late fees apply.\n\n4. Liability\nCapped at fees.\n"
    seg = segment(text)
    assert heads(seg) == [("1.1", "Payment"), ("1.2", "Payment"), ("4", "Liability")]
    c = seg.clauses[2]
    assert text[c.start:c.end] == "Capped at fees."


def test_section_style_with_alpha_subclauses():
    seg = segment("AGREEMENT\n\nSection 7. Termination\n(a) Either party may end it.\n(b) Effect of ending.\n")
    assert [c.number for c in seg.clauses] == ["7(a)", "7(b)"]
    assert all(c.heading == "Termination" for c in seg.clauses)


def test_article_heading_and_nice_title():
    seg = segment("AGREEMENT\n\nARTICLE IV - FEES AND PAYMENT\n4.1 Invoices are due net 30.\n")
    assert seg.clauses[0].heading == "Fees and Payment" and seg.clauses[0].number == "4.1"


def test_inline_heading_split_from_body():
    text = "AGREEMENT\n\n9. Limitation of Liability. Except for fraud, liability is capped.\n9.2. Second part.\n"
    seg = segment(text)
    assert seg.clauses[0].heading == "Limitation of Liability"
    assert seg.clauses[0].text == "Except for fraud, liability is capped."
    assert seg.clauses[1].heading == "Limitation of Liability"


def test_caps_headings_and_one_paragraph_per_line():
    seg = segment("AGREEMENT\n\nCONFIDENTIALITY\nFirst sentence.\nSecond paragraph.\n")
    assert [c.text for c in seg.clauses] == ["First sentence.", "Second paragraph."]
    assert seg.clauses[0].heading == "Confidentiality"


def test_wrapped_line_is_continuation():
    text = "AGREEMENT\n\n1. Payment\nCustomer shall pay all invoices\nwithin thirty days.\n"
    seg = segment(text)
    assert len(seg.clauses) == 1 and seg.clauses[0].text.endswith("within thirty days.")


def test_all_caps_long_paragraph_is_not_a_heading():
    text = ("AGREEMENT\n\n3. Warranties\nEXCEPT AS EXPRESSLY PROVIDED IN THIS AGREEMENT, VENDOR DISCLAIMS ALL IMPLIED "
            "WARRANTIES, INCLUDING MERCHANTABILITY.\n")
    seg = segment(text)
    assert seg.clauses[0].heading == "Warranties" and seg.clauses[0].text.startswith("EXCEPT")


def test_paren_numbering_and_markdown_headings():
    seg = segment("AGREEMENT\n\n2) Payment\n2.1) Net 30.\n\n# Insurance\nCoverage of $2 million.\n")
    assert heads(seg) == [("2.1", "Payment"), ("", "Insurance")]


def test_order_form_key_values_and_kind():
    text = "ORDER FORM\n\nCustomer: Brightwater\nPayment Terms: Net 45\n"
    seg = segment(text)
    assert seg.kind == "order_form"
    assert [c.label for c in seg.clauses] == ["Customer", "Payment Terms"]


def test_definitions_collected():
    text = 'AGREEMENT\n\n1. Definitions\n1.1 "Liability Cap" means twice the fees.\n1.2 "Fees" means money.\n'
    assert segment(text).definitions == {"Liability Cap": "twice the fees.", "Fees": "money."}


def test_detect_kind_from_title():
    assert detect_kind("DATA PROCESSING AGREEMENT", "") == "dpa"
    assert detect_kind("Master Services Agreement", "including all Order Forms") == "msa"


def test_sentences_keep_decimals_and_offsets():
    s = sentences("Interest at 1.5% per month. Tessavel, Inc. agrees.", 100)
    assert [x[0] for x in s] == ["Interest at 1.5% per month.", "Tessavel, Inc. agrees."]
    assert s[1][1] == 128


def test_every_truth_span_is_a_segmented_clause(corpus):
    """All six numbering styles in the corpus segment to exactly the labeled paragraph spans."""
    for c in corpus:
        seg = segment(c.text)
        spans = {(x.start, x.end) for x in seg.clauses}
        for cl in c.clauses.values():
            for a, b in cl.spans:
                assert (a, b) in spans, (c.id, cl.type, c.text[a:b])
        assert seg.kind == c.kind
