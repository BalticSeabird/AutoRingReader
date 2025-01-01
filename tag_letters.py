_tag_letters_tuple = (
    "AAA",
    "AAB",
    "AAC",
    "AAD",
    "AAF",
    "AAH",
    "AAK",
    "AAL",
    "AAN",
    "AAR",
    "AAS",
    "AAT",
    "AAV",
    "AAX",
    "AAZ",
    "ABC",
    "ABD",
    "ABH",
    "ABJ",
    "ABK",
    "ABL",
    "ABN",
    "ABP",
    "ABR",
    "ABS",
    "ABT",
    "ABV",
    "ABZ",
    "ACA",
    "ACB",
    "ACC",
    "ACD",
    "ACE",
    "ACF",
    "ACG",
    "ACH",
    "ACJ",
    "ACK",
    "ACL",
    "ACN",
    "ACP",
    "ACR",
    "ACS",
    "ACT",
    "ACV",
    "ACX",
    "ACZ",
    "ADA",
    "ADB",
    "ADC",
    "ADD",
    "ADE",
    "ADG",
    "ADJ",
    "ADK",
)


s0 = set()
s1 = set()
s2 = set()
for s in _tag_letters_tuple:
    assert len(s) == 3
    assert s.isalpha()
    s0.add(s[0])
    s1.add(s[1])
    s2.add(s[2])

_letters_pos_0 = "[" + "".join(sorted(s0)) + "]"
_letters_pos_1 = "[" + "".join(sorted(s1)) + "]"
_letters_pos_2 = "[" + "".join(sorted(s2)) + "]"

#print(letters_pos_0)
#print(letters_pos_1)
#print(letters_pos_2)