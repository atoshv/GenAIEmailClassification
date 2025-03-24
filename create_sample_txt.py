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

import random
from datetime import datetime

# @@AUTHOR:Atosh Veer@@ 
def generate_email():
    if random.random() > 0.5:  # 50% chance for each type
        # Money movement email
        return f"""Subject: Transfer Request {datetime.now().strftime('%d-%m-%Y')}

Dear Team,

Please process this payment:

Amount: ${random.randint(1,20)},{random.randint(100,999):03d}
Account: ACCT-{random.randint(10000,99999)}
Date: {datetime.now().strftime('%d%b%y').upper()}
Reference: {random.choice(['Loan', 'Invoice', 'Settlement'])} Payment

Regards,
Client Services
"""
    else:
        # Adjustment email
        return f"""Subject: Rate Adjustment Request

Dear Loan Team,

Requesting rate modification:

Loan ID: LN-{random.randint(100000,999999)}
Current Rate: {random.uniform(3.0,8.0):.2f}%
Requested Rate: {random.uniform(2.0,7.5):.2f}%
Effective Date: {datetime.now().strftime('%Y-%m-%d')}

Thank you,
Account Management
"""

with open("sample_email.txt", "w") as f:
    f.write(generate_email())
print("Generated sample_email.txt")