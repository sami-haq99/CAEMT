#read a .tsv file
import sys
import csv
import argparse
import os

def fix_broken_row(row):
    if None in row and row.get('method') == 'news':
        # Move 'news' to domain_name
        row['domain_name'] = 'news'
        # Move the None key value (a list like ['ESA']) to 'method'
        row['method'] = row[None][0] if isinstance(row[None], list) else row[None]
        # Remove the bad None key
        del row[None]
        print("Fixed row:", row)
    return row


#read a tsv file and return a list of dictionaries
def read_tsv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        data = [row for row in reader]
    print("Structure:", reader.fieldnames)

    #print the unique value of 'method' column
    if 'method' in reader.fieldnames:
        unique_methods = set(row['method'] for row in data if 'method' in row)
        print("Unique methods:", unique_methods)
    # 
    #
    source_langs = set(row['source_lang'] for row in data if 'source_lang' in row)
    target_langs = set(row['target_lang'] for row in data if 'target_lang' in row)
    print("Unique source languages:", source_langs)
    print("Unique target languages:", target_langs)

    # extract the records with 'source_lang' == 'en' and 'target_lang' == 'ru' or 'zh_RU' 
    filtered_data_broken = [fix_broken_row(row) for row in data if (row.get('source_lang') == 'en' and (row.get('target_lang') == 'ru_RU')) and (row.get('method') != 'ESA')]
    print("Filtered data count:", len(filtered_data_broken))

    
    # extract the records with 'source_lang' == 'en' and 'target_lang' == 'ja' or 'ja_JP'
    #filtered_data = [row for row in data if row.get('source_lang') == 'en' and (row.get('target_lang') == 'ru' or row.get('target_lang') == 'ru_RU') and (row.get('method') == 'ESA')]
    #save the filtered data to a csv file
    #print("Filtered data count with method 'ESA':", len(filtered_data))
    #merged_data = filtered_data_broken + filtered_data
   
    #print("filtered_data_broken:", len(filtered_data_broken))
    #print("filtered_data:", len(filtered_data))
    #print("merged_data:", len(merged_data))

    filtered_file_path = 'filtered_data_en_ru-broken.csv'
    with open(filtered_file_path, 'w', newline='') as filtered_file:
        writer = csv.DictWriter(filtered_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(filtered_data_broken)

    


# Main function to handle command line arguments
def main():
    
    data = read_tsv('mteval-task1-test25.tsv')
    if data:
        print("Data read successfully.")
    else:
        print("No data found in the file.")

if __name__ == "__main__":
    main()
    # Uncomment the line below to read a different file
    # data = read_tsv('mteval-task1-test25.tsv')    
