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

from email_classification import process_email
import glob
import json

# @@AUTHOR:Atosh Veer@@ 
def process_email_batch(input_folder, output_file):
    results = []
    files = glob.glob(f"{input_folder}/*.*") 
    
    for file in files:
        if file.lower().endswith(('.pdf', '.eml', '.txt', '.doc', '.docx')):
            result = process_email(file)
            if result:
                results.extend(result)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)