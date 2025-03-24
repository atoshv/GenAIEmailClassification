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

# create_sample_eml.py
import random
from datetime import datetime

# @@AUTHOR:Atosh Veer@@ 
def create_eml_file(file_path, content):
    """
    Creates an EML file with the given content.
    Args:
        file_path (str): Path to save the EML file.
        content (str): Content to write to the EML file.
    """
    with open(file_path, "w") as f:
        f.write(content)
    print(f"EML file created successfully: {file_path}")

def generate_money_movement_email():
    """Generate a realistic money movement email with random but valid data"""
    amount = f"${random.randint(1, 50)},{random.randint(100, 999):03d}"
    date = datetime.now().strftime("%d%b%y").upper()
    account = random.choice(["Atosh", "CANTO R FITZGERA", "ACCT-78912", "Loan-45632"])
    trans_type = random.choice(["Principal", "Interest", "Principal + Interest"])
    
    return f"""From: {account.replace(' ', '')} <client{random.randint(1, 100)}@example.com>
To: Loan Servicing Team <loans@bank.com>
Subject: Request for Money Movement {random.choice(['Inbound', 'Outbound'])}

Dear Team,

I would like to request a money movement for the following transaction:

- Amount: {amount}
- Type: {trans_type}
- Date: {date}
- Account: {account}
- Reference: {random.choice(['Loan Payment', 'Fee Settlement', 'Advance Payment'])}

Please process this request at your earliest convenience.

Best regards,
{account.split()[0] if ' ' in account else account}
"""

# Create the EML file with randomized but realistic content
email_content = generate_money_movement_email()
create_eml_file("sample_email.eml", email_content.strip())