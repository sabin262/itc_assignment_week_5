from pathlib import Path
from textwrap import fill


OUTPUT_DIR = Path("sample_leases")


LEASES = [
    {
        "filename": "dense_legal_lease_e_ashbourne_court.txt",
        "title": "FULL FORM RESIDENTIAL TENANCY AGREEMENT - ASHBOURNE COURT",
        "pages": 56,
        "landlord": "Ashbourne Court Holdings Limited",
        "tenant": "Helena Ward and Marcus Llewellyn",
        "property": "Maisonette 8, Ashbourne Court, 3 Belvedere Crescent, Bath BA1 5QY",
        "start": "1 June 2026",
        "end": "31 May 2028",
        "rent": "GBP 2,050",
        "due": "the third calendar day of each month",
        "deposit": "GBP 2,365",
        "notice": "not less than three calendar months' prior written notice",
        "access_notice": "not less than forty eight hours' prior written notice",
        "pets": "no animal shall be kept at the Premises save for one small dog approved by the Landlord in writing",
        "guests": "no guest may reside for more than ten nights in any calendar month without prior written consent",
        "noise": "no audible nuisance, amplified sound, or unreasonable domestic disturbance shall continue after 9:45 pm",
        "unusual": [
            "the Tenant shall not use abrasive products on the Grade II listed stone fireplace surround",
            "the Tenant shall permit quarterly inspection of sash window cords due to the listed-building maintenance programme",
            "the Tenant shall keep the cellar ventilation grille unobstructed at all times",
            "the Tenant shall not place planters, bicycles, or furniture upon the shared Georgian entrance steps",
            "the Tenant shall notify the Landlord before operating any portable air-conditioning unit over 900 watts",
        ],
    },
    {
        "filename": "dense_legal_lease_f_northgate_harbour.txt",
        "title": "DEED OF RESIDENTIAL LEASE AND TENANCY COVENANTS - NORTHGATE HARBOUR",
        "pages": 58,
        "landlord": "Northgate Harbour Residential REIT plc",
        "tenant": "Eleanor Singh and Tomasz Kowalski",
        "property": "Apartment 2304, Northgate Harbour Tower, 11 Dockmaster Lane, Liverpool L3 1HG",
        "start": "10 July 2026",
        "end": "9 July 2029",
        "rent": "GBP 2,425",
        "due": "the fifth business day of each month",
        "deposit": "GBP 2,798",
        "notice": "not less than four calendar months' written notice",
        "access_notice": "not less than twenty four hours' written notice",
        "pets": "no pet, animal, bird, reptile, or other creature may be kept without express written licence",
        "guests": "overnight visitors may not remain for more than fourteen aggregate nights in any rolling ninety day period",
        "noise": "quiet enjoyment restrictions prohibit parties, amplified music, and balcony gatherings after 10:00 pm",
        "unusual": [
            "the Tenant must participate in annual cladding and balcony safety access appointments",
            "the Tenant must not tamper with the mechanical ventilation heat recovery system",
            "the Tenant must maintain an emergency grab-bag because the building is within a managed dock flood zone",
            "the Tenant must not store lithium e-bike batteries inside the apartment overnight",
            "the Tenant must use the resident portal to book goods-lift slots for bulky deliveries",
        ],
    },
]


HEADINGS = [
    "Recitals, Construction, and Operative Demise",
    "Definitions and Rules of Interpretation",
    "Term Certain, Contractual Expiry, and No Implied Renewal",
    "Rent Covenant, Time of Payment, and Method of Discharge",
    "Deposit, Statutory Protection, Permitted Deductions, and Accounting",
    "Interest, Default Administration, and Non-waiver of Rent",
    "Permitted Occupiers, Capacity, and Residential User",
    "Condition Precedent, Inventory, and Evidential Photographs",
    "Tenant's Positive Covenants as to Occupation",
    "Tenant's Negative Covenants and Prohibited Acts",
    "Quiet Enjoyment, Nuisance, and Neighbouring Premises",
    "Pets, Animals, Assistance Animals, and Additional Licence Conditions",
    "Visitors, Guests, Lodgers, and Deemed Occupation",
    "Repairs Reserved to the Landlord",
    "Tenant Repair Liability and Damage Occasioned by Default",
    "Notice of Defect, Mitigation of Loss, and Emergency Procedure",
    "Access, Inspection, Works, and Contractor Attendance",
    "Alterations, Additions, Fixtures, and Reinstatement",
    "Decoration, Fastenings, Floor Coverings, and Fabric Protection",
    "Utilities, Metering, Charges, and Apportionments",
    "Heating, Ventilation, Condensation, and Mould Prevention",
    "Electrical Equipment, Appliances, and Portable Devices",
    "Fire Safety, Means of Escape, and Alarm Testing",
    "Insurance, Tenant Belongings, and Exclusions",
    "Assignment, Subletting, Sharing, and Parting with Possession",
    "Business Use, Home Working, and Commercial Restrictions",
    "Common Parts, Estate Regulations, and Managing Agent Directions",
    "Parking, Cycle Storage, Deliveries, and Access Devices",
    "Balconies, Terraces, External Areas, and Window Displays",
    "Refuse, Recycling, Pest Prevention, and Hygiene",
    "Data Protection, Smart Systems, and Resident Portal Use",
    "Notices, Deemed Service, and Communications Protocol",
    "Notice to Vacate, Check-out, and Yielding Up",
    "Early Surrender, Replacement Tenant, and Mitigation",
    "Abandonment, Extended Absence, and Security Checks",
    "Breach, Remedy Periods, and Reservation of Rights",
    "Statutory Compliance, Consents, and Superior Interests",
    "Listed Building, Conservation, or High-rise Specific Covenants",
    "Flood, Severe Weather, Water Escape, and Building Resilience",
    "Dispute Handling, Evidence, and Without Prejudice Discussions",
    "Entire Agreement, Variation, Severance, and Counterparts",
    "Schedule of Financial Terms",
    "Schedule of Tenant Day-to-day Obligations",
    "Schedule of Landlord Operational Obligations",
    "Schedule of Non-standard and Site-specific Clauses",
    "Schedule of Inventory Assumptions",
    "Schedule of Move-out Cleaning and Return Standard",
    "Execution, Acknowledgements, and Plain-English Summary",
]


LEGAL_CONNECTORS = [
    "provided always that",
    "without prejudice to the generality of the foregoing",
    "for the avoidance of doubt",
    "save where expressly stated otherwise",
    "subject to the contrary being required by statute",
    "and in each case acting reasonably where consent is required",
]


def paragraph(text: str) -> str:
    return fill(" ".join(text.split()), width=96)


def legal_page(lease: dict[str, object], page_number: int, heading: str) -> str:
    connector = LEGAL_CONNECTORS[(page_number - 1) % len(LEGAL_CONNECTORS)]
    unusual = lease["unusual"][(page_number - 1) % len(lease["unusual"])]
    second_unusual = lease["unusual"][page_number % len(lease["unusual"])]

    clauses = [
        (
            f"{page_number}.1 This clause shall be construed as part of the tenancy agreement made "
            f"between {lease['landlord']} as landlord and {lease['tenant']} as tenant in respect of "
            f"{lease['property']}, and the demise hereby granted is for the fixed term commencing "
            f"on {lease['start']} and expiring on {lease['end']}, {connector}, no estate, interest, "
            f"licence, easement, right, or indulgence shall be deemed to arise other than as expressly "
            f"conferred by this Agreement."
        ),
        (
            f"{page_number}.2 The Tenant covenants with the Landlord to pay the rent of {lease['rent']} "
            f"monthly in advance on {lease['due']} without deduction, set-off, counterclaim, abatement, "
            f"or suspension except where such restriction is prohibited by law, and further acknowledges "
            f"that the deposit of {lease['deposit']} is held as security for the due performance and "
            f"observance of the Tenant's covenants, conditions, stipulations, and agreements contained "
            f"herein."
        ),
        (
            f"{page_number}.3 The Tenant shall not, and shall procure that all occupiers, invitees, "
            f"contractors, licensees, and visitors shall not, use the Premises otherwise than as a "
            f"private residential dwelling; cause nuisance, annoyance, obstruction, damage, hazard, "
            f"or interference; or do any act by reason whereof any policy of insurance may be vitiated, "
            f"rated, avoided, or rendered subject to additional premium."
        ),
        (
            f"{page_number}.4 The Tenant's ordinary obligations include keeping the Premises clean, "
            f"heated, ventilated, secured, and reasonably protected from foreseeable deterioration; "
            f"promptly notifying the Landlord of defect, disrepair, infestation, leak, failure of services, "
            f"or suspected hazard; and taking immediate reasonable steps to mitigate loss pending attendance "
            f"by the Landlord, managing agent, contractor, utility provider, or emergency authority."
        ),
        (
            f"{page_number}.5 The Landlord covenants to keep in repair the structure, exterior, installations "
            f"for the supply of water, gas, electricity, sanitation, space heating, and water heating so far "
            f"as required by applicable residential landlord obligations, and to use reasonable endeavours "
            f"to procure repairs after notice, subject to access, parts availability, contractor attendance, "
            f"superior landlord requirements, insurance conditions, and force majeure events."
        ),
        (
            f"{page_number}.6 Save in emergency, suspected abandonment, urgent safety works, water escape, "
            f"fire risk, or circumstances where immediate entry is reasonably necessary to prevent loss or "
            f"injury, the Landlord shall give {lease['access_notice']} before entering the Premises for "
            f"inspection, repair, valuation, viewing, statutory compliance, testing, or works required by "
            f"this Agreement or by any competent authority."
        ),
        (
            f"{page_number}.7 The following practical restrictions are material terms: {lease['pets']}; "
            f"{lease['guests']}; and {lease['noise']}. Any permission granted in respect of pets, guests, "
            f"noise, storage, parking, deliveries, or alterations shall be personal, revocable where breached, "
            f"non-transferable, and shall not constitute a variation unless confirmed in writing as such."
        ),
        (
            f"{page_number}.8 The parties agree that the non-standard provisions applicable to this Premises "
            f"include that {unusual}; and further that {second_unusual}. Such provisions are not boilerplate "
            f"terms and are included having regard to the age, location, construction, management regime, "
            f"amenities, insurance arrangements, or building safety requirements of the Premises."
        ),
        (
            f"{page_number}.9 The Tenant shall give {lease['notice']} if the Tenant intends to vacate at or "
            f"after the contractual expiry, and shall yield up the Premises with vacant possession, cleaned, "
            f"cleared of personal items, with keys and access devices returned, and with fair wear and tear "
            f"excepted. Failure to comply may entitle the Landlord to claim documented loss, cleaning charges, "
            f"replacement key charges, or other recoverable sums lawfully due."
        ),
        (
            f"{page_number}.10 No failure, delay, leniency, acceptance of rent, inspection, correspondence, "
            f"or partial enforcement by the Landlord shall amount to a waiver of any breach unless expressly "
            f"confirmed in writing. Any invalidity, unenforceability, or statutory modification affecting one "
            f"provision shall not affect the remaining provisions, which shall continue so far as lawful and "
            f"capable of taking effect according to their tenor."
        ),
    ]

    page = f"PAGE {page_number} OF {lease['pages']}\n{heading.upper()}\n\n"
    page += "\n\n".join(paragraph(clause) for clause in clauses)
    return page


def make_lease(lease: dict[str, object]) -> str:
    pages = int(lease["pages"])
    blocks = [
        str(lease["title"]),
        "",
        "DENSE SAMPLE DOCUMENT FOR TESTING A LEASE SUMMARISATION SYSTEM. THIS IS NOT LEGAL ADVICE.",
        "",
        paragraph(
            f"THIS AGREEMENT is made between {lease['landlord']} and {lease['tenant']} in relation "
            f"to {lease['property']}. It records a private residential tenancy beginning on "
            f"{lease['start']} and ending on {lease['end']}, at a monthly rent of {lease['rent']} "
            f"payable on {lease['due']}, with a security deposit of {lease['deposit']} and a notice "
            f"period to vacate of {lease['notice']}."
        ),
    ]

    for page_number in range(1, pages + 1):
        heading = HEADINGS[(page_number - 1) % len(HEADINGS)]
        blocks.append(legal_page(lease, page_number, heading))
        blocks.append("\f")

    blocks.append(
        paragraph(
            f"Plain-English summary for testing only: {lease['tenant']} rents {lease['property']} "
            f"from {lease['landlord']} from {lease['start']} until {lease['end']}. Rent is "
            f"{lease['rent']} due on {lease['due']}. The deposit is {lease['deposit']}. The tenant "
            f"must give {lease['notice']} to vacate, comply with detailed restrictions on guests, "
            f"pets, noise, maintenance, access, and site-specific clauses, and return the premises "
            f"clean and vacant at the end of the term. The landlord must maintain key systems, handle "
            f"repairs, and give required access notice except in emergencies."
        )
    )
    return "\n\n".join(blocks)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for lease in LEASES:
        path = OUTPUT_DIR / str(lease["filename"])
        path.write_text(make_lease(lease), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
