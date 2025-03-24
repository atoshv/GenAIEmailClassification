"""
Email Classification System for Financial Requests
Copyright (C) 2024 atoshveer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
__author__ = "Atosh Veer <veeratosh@gmail.com>"
__license__ = "GPL-3.0"

from fpdf import FPDF
import random
from datetime import datetime

# @@AUTHOR:Atosh Veer@@ 
def generate_random_request():
    request_type = random.choice(['MoneyMovement', 'Adjustment'])
    if request_type == 'MoneyMovement':
        amount = f"${random.randint(5, 50)},{random.randint(100, 999):03d}"
        date = datetime.now().strftime("%d%b%y").upper()
        account = random.choice(["CANTO R FITZGERA", "ACME CORP", "GLOBAL INVEST"])
        return f"""
Subject: Funds Transfer Request

Dear Team,

Please process this {random.choice(['inbound', 'outbound'])} transaction:

- Amount: {amount}
- Account: {account}
- Value Date: {date}
- Reference: {random.choice(['Loan Payment', 'Invoice Settlement'])}

Regards,
{account.split()[0]}
"""
    else:
        loan_id = f"LN-{random.randint(100000, 999999)}"
        current = f"{random.uniform(3.0, 8.0):.2f}%"
        new = f"{float(current[:-1]) - random.uniform(0.5, 2.0):.2f}%"
        return f"""
Subject: Rate Adjustment Request

Dear Loan Team,

Requesting rate change:

- Loan ID: {loan_id}
- Current Rate: {current}
- Requested Rate: {new}
- Effective: {datetime.now().strftime("%d-%b-%Y")}

Sincerely,
Client Services
"""

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
content = generate_random_request()
pdf.multi_cell(0, 10, content.strip())
pdf.output("financial_request.pdf")
print("Generated financial_request.pdf")