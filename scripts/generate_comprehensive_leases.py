from pathlib import Path
from textwrap import fill


OUTPUT_DIR = Path("sample_leases")


LEASES = [
    {
        "filename": "comprehensive_lease_c_riverside_lofts.txt",
        "title": "COMPREHENSIVE RESIDENTIAL LEASE AGREEMENT - RIVERSIDE LOFTS",
        "pages": 48,
        "landlord": "Riverside Quarter Residential LLP",
        "tenant": "Amelia Chen and Noah Brooks",
        "property": "Unit 1206, Riverside Lofts, 42 Merchant Walk, Leeds LS1 4PD",
        "start": "1 March 2026",
        "end": "28 February 2027",
        "rent": "GBP 1,875",
        "due": "the first business day of each month",
        "deposit": "GBP 2,160",
        "notice": "60 days written notice",
        "access_notice": "at least twenty four hours written notice",
        "pet_rule": "one neutered indoor cat is permitted after registration with building management",
        "guest_rule": "overnight guests may stay for up to fourteen nights in any rolling sixty day period",
        "noise_rule": "quiet hours run from 10:00 pm to 7:00 am Sunday through Thursday and from 11:00 pm to 8:00 am on Friday and Saturday",
        "unusual": [
            "the tenant must keep the smart thermostat connected so heating efficiency reports can be produced",
            "the tenant must use the RFID bicycle room tag and report any lost tag within twenty four hours",
            "the tenant may use the roof garden but must not remove community-grown herbs except on posted harvest days",
            "the tenant must keep a flood preparation kit in the utility cupboard because the building sits near the river",
            "the tenant may reserve an electric vehicle charging bay twice per week subject to building availability",
        ],
    },
    {
        "filename": "comprehensive_lease_d_rowan_mews.txt",
        "title": "COMPREHENSIVE RESIDENTIAL LEASE AGREEMENT - ROWAN MEWS TOWNHOUSE",
        "pages": 52,
        "landlord": "Oak & Lantern Property Company Ltd",
        "tenant": "Priya Shah and Daniel Morgan",
        "property": "19 Rowan Mews, Cambridge CB4 1ZX",
        "start": "15 April 2026",
        "end": "14 April 2028",
        "rent": "GBP 2,250",
        "due": "the fifth calendar day of each month",
        "deposit": "GBP 2,596",
        "notice": "90 days written notice",
        "access_notice": "at least forty eight hours written notice",
        "pet_rule": "no pets are allowed except a registered assistance animal or a small caged pet approved in writing",
        "guest_rule": "guests may stay for up to seven consecutive nights and no more than twenty nights per calendar quarter",
        "noise_rule": "musical instruments, amplified sound, and garden gatherings must stop by 9:30 pm every day",
        "unusual": [
            "the tenant must use only conservation-safe cleaning products on the original timber staircase",
            "the tenant must allow an annual chimney sweep even if the fireplace is not used",
            "the tenant must maintain the rain garden and water butt overflow system during spring and summer",
            "the tenant may use the electric vehicle charger but must reimburse metered charging costs monthly",
            "the tenant may keep an upright piano, but practice is limited to two hours per day before 8:00 pm",
        ],
    },
]


SECTIONS = [
    "Parties, Property, and Grant of Tenancy",
    "Lease Term, Renewal Discussions, and Holding Over",
    "Rent Amount, Due Date, and Payment Method",
    "Late Payment, Returned Payment, and Administrative Charges",
    "Security Deposit, Deductions, and Deposit Return Process",
    "Permitted Occupants and Household Information",
    "Use of Property and Residential Purpose",
    "Move-in Condition, Inventory, and Photographs",
    "Utilities, Council Tax, Broadband, and Meter Readings",
    "Tenant Cleaning, Waste, Recycling, and Pest Prevention",
    "Routine Maintenance and Tenant Reporting Duties",
    "Landlord Repairs, Response Times, and Essential Services",
    "Emergency Repairs and After-hours Contact Procedure",
    "Landlord Access, Inspection, and Entry Notice",
    "Pets, Assistance Animals, and Animal Damage",
    "Guests, Occupancy Limits, and Short-term Visitors",
    "Noise, Nuisance, Neighbours, and Anti-social Conduct",
    "Smoking, Vaping, Candles, and Fire Safety",
    "Appliances, Fixtures, Furniture, and Supplied Equipment",
    "Alterations, Decorations, Fastenings, and Improvements",
    "Keys, Locks, Access Devices, and Building Fobs",
    "Parking, Bicycle Storage, Deliveries, and Shared Areas",
    "Balconies, Gardens, Exterior Areas, and Window Coverings",
    "Water Leaks, Condensation, Ventilation, and Mould Prevention",
    "Heating, Cooling, Energy Use, and Smart Devices",
    "Internet-connected Devices, Data Notices, and Security",
    "Insurance, Personal Property, and Tenant Risk",
    "Prohibited Conduct, Illegal Activity, and Hazardous Materials",
    "Subletting, Assignment, Lodgers, and Business Use",
    "Repairs Caused by Tenant, Guests, or Negligence",
    "Notice to Vacate, End-of-term Procedure, and Check-out",
    "Early Termination, Break Discussions, and Replacement Tenant",
    "Abandonment, Extended Absence, and Property Checks",
    "Rent Review, Fee Limitations, and Written Variations",
    "Complaints, Dispute Resolution, and Record Keeping",
    "Service of Notices and Approved Communication Channels",
    "Compliance with Building Rules and Managing Agent Policies",
    "Health, Safety, Alarms, and Statutory Testing",
    "Fire Doors, Escape Routes, and Emergency Planning",
    "Damp, Flood, Severe Weather, and Property Protection",
    "Waste Rooms, Communal Storage, and Cleanliness Standards",
    "Deliveries, Parcels, Contractors, and Visitor Access",
    "Confidentiality, Privacy, and Data Handling",
    "Unusual Clauses and Site-specific Requirements",
    "Schedule 1 - Financial Terms",
    "Schedule 2 - Inventory and Condition Summary",
    "Schedule 3 - Tenant Obligations Summary",
    "Schedule 4 - Landlord Obligations Summary",
    "Schedule 5 - Building and Neighbourhood Rules",
    "Schedule 6 - Move-out Cleaning Standard",
    "Schedule 7 - Notices, Signatures, and Acknowledgements",
    "Execution Page and Practical Plain-English Summary",
]


def paragraph(text: str) -> str:
    return fill(" ".join(text.split()), width=92)


def make_page(lease: dict[str, object], page_number: int, heading: str) -> str:
    unusual = lease["unusual"][(page_number - 1) % len(lease["unusual"])]
    page = f"PAGE {page_number} OF {lease['pages']}\n{heading.upper()}\n\n"
    page += paragraph(
        f"This section forms part of the residential lease between the landlord, "
        f"{lease['landlord']}, and the tenant, {lease['tenant']}, for {lease['property']}. "
        f"The tenancy runs from {lease['start']} to {lease['end']}. The parties agree that "
        f"each obligation in this section must be read together with the rent, deposit, "
        f"notice, repair, access, and conduct provisions elsewhere in this agreement."
    )
    page += "\n\n"
    page += paragraph(
        f"For this topic, the tenant must act reasonably, promptly report issues, protect "
        f"the property from avoidable damage, and avoid conduct that interferes with neighbours "
        f"or building operations. The landlord must maintain the property in a habitable condition, "
        f"arrange repairs that are the landlord's responsibility, keep appropriate records, and "
        f"give {lease['access_notice']} before ordinary non-emergency entry. Emergency access may "
        f"occur without prior notice when needed to prevent injury, serious loss, or material damage."
    )
    page += "\n\n"
    page += paragraph(
        f"The monthly rent is {lease['rent']} and is due on {lease['due']}. The security deposit is "
        f"{lease['deposit']}. The required notice period to vacate is {lease['notice']}. The tenant "
        f"must follow these financial and notice terms even when this page discusses a practical "
        f"housekeeping matter, because payment, communication, and access obligations are continuing "
        f"terms of the lease."
    )
    page += "\n\n"
    page += paragraph(
        f"Specific operating rules for this lease include the following: {lease['pet_rule']}; "
        f"{lease['guest_rule']}; and {lease['noise_rule']}. A site-specific clause also provides that "
        f"{unusual}. These provisions are included because the property has particular building, "
        f"neighbourhood, or maintenance requirements that may not appear in a shorter standard lease."
    )
    page += "\n\n"
    page += paragraph(
        f"If a dispute arises about this section, the parties should first review written notices, "
        f"inspection photographs, payment records, maintenance reports, and any managing agent emails. "
        f"No waiver is created merely because one party delays enforcing a term. Any amendment must be "
        f"made in writing and signed or confirmed by both parties through the approved notice channel."
    )
    return page


def make_lease(lease: dict[str, object]) -> str:
    pages = int(lease["pages"])
    content = [
        str(lease["title"]),
        "",
        "SAMPLE DOCUMENT FOR TESTING A LEASE SUMMARISATION SYSTEM. THIS IS NOT LEGAL ADVICE.",
        "",
    ]
    for page_number in range(1, pages + 1):
        heading = SECTIONS[(page_number - 1) % len(SECTIONS)]
        content.append(make_page(lease, page_number, heading))
        content.append("\f")
    content.append(
        paragraph(
            f"Plain-English summary: {lease['tenant']} rents {lease['property']} from "
            f"{lease['landlord']} from {lease['start']} to {lease['end']}. Rent is "
            f"{lease['rent']} due on {lease['due']}, the deposit is {lease['deposit']}, and "
            f"the notice period to vacate is {lease['notice']}. The tenant has detailed duties "
            f"around maintenance, guests, noise, access devices, and site-specific rules. The "
            f"landlord has detailed duties around repairs, access notice, safety checks, and "
            f"record keeping."
        )
    )
    return "\n\n".join(content)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for lease in LEASES:
        path = OUTPUT_DIR / str(lease["filename"])
        path.write_text(make_lease(lease), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
