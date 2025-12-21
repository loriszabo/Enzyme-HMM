import urllib.request
import re
import pandas as pd

# --- CONFIGURATION ---
IMGT_URL = "https://www.imgt.org/download/GENE-DB/IMGTGENEDB-ReferenceSequences.fasta-nt-WithoutGaps-F+ORF+inframeP"
SPECIES = "Homo sapiens"
GENE_TYPE = "IGHV"  # We only want Heavy Chain Variable genes for the bridge

def fetch_and_parse_germlines():
    print(f"Downloading reference sequences from IMGT...\n{IMGT_URL}")
    
    # 1. Download the data
    with urllib.request.urlopen(IMGT_URL) as response:
        fasta_data = response.read().decode('utf-8')

    # 2. Parse FASTA (Manual parsing to avoid Biopython dependency)
    sequences = []
    current_header = None
    current_seq = []

    print("Parsing and filtering...")
    for line in fasta_data.splitlines():
        line = line.strip()
        if line.startswith(">"):
            # Save previous sequence if it matched criteria
            if current_header and current_seq:
                full_seq = "".join(current_seq)
                # Filter: Must be Human AND IGHV
                if SPECIES in current_header and GENE_TYPE in current_header:
                    sequences.append(full_seq)
            
            # Reset
            current_header = line
            current_seq = []
        else:
            current_seq.append(line)
            
    # Capture the last entry
    if current_header and current_seq:
        full_seq = "".join(current_seq)
        if SPECIES in current_header and GENE_TYPE in current_header:
            sequences.append(full_seq)

    print(f"Found {len(sequences)} unique {SPECIES} {GENE_TYPE} germline sequences.")
    
    # 3. Save to CSV for your generator
    df = pd.DataFrame({"germline_seq": sequences})
    
    # Filter out extremely short fragments (artifacts)
    df = df[df['germline_seq'].str.len() > 200]
    
    output_file = "human_ighv_germlines.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    
    return df

if __name__ == "__main__":
    df = fetch_and_parse_germlines()
    print(df.head())