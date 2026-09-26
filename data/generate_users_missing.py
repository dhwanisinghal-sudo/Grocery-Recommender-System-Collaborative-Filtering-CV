"""
Generates the missing U001-U050 rows for data/users_new.csv (Gap #7).

users_new.csv, as it existed before this script, only covered U051-U150
(100 rows) -- U001-U050 appear as rated users in user_ratings.csv but had
no demographic profile anywhere in the repo. app.py does not read this
file at all (it only reads user_ratings.csv for recommendations), so this
gap was cosmetic/documentation-only, not a functional bug -- but the file
claimed to be user metadata while covering only 2/3 of the user base.

This script is, like data/generate_ratings.py, FULLY SYNTHETIC: there is
no real person behind any of U001-U050's name, age, gender, location, or
cluster label below. It exists only to make users_new.csv internally
consistent (i.e. it now actually covers all 150 users that appear in
user_ratings.csv), not to represent real demographic data.

Method (random_state=42, advanced in user_id order U001..U050):
  1. gender ~ Bernoulli(p=0.52 male), matching the observed 52/48 split
     in the existing U051-U150 rows.
  2. name ~ drawn from the exact first-name pools (split by gender) and
     last-name pool already used in U051-U150, so new names are stylistically
     indistinguishable from the existing ones rather than introducing a
     new naming convention.
  3. age ~ round(clip(Normal(38.58, 12.11), 18, 60)), matching the observed
     mean/stdev of the existing 100 rows.
  4. location ~ categorical, weighted to match the existing 10-city
     distribution (Bangalore 18%, Jaipur 17%, Ahmedabad 11%, Delhi 9%,
     Surat 9%, Chennai 8%, Pune 8%, Kolkata 8%, Hyderabad 7%, Mumbai 5%).
  5. cluster ~ categorical, weighted to match the existing 6-cluster
     distribution (health_conscious 27%, tea_coffee_fan 21%, home_chef 14%,
     snack_lover 14%, busy_professional 13%, family_shopper 11%).
  6. preferred_categories is NOT independently drawn -- in the existing
     data every user in a given cluster has the *same* fixed category
     string (a deterministic lookup table, not a distribution), so step 6
     just applies that same lookup table to the cluster drawn in step 5.

Run from the repo root:
    python data/generate_users_missing.py
Writes data/users_new_completed.csv (150 rows, U001-U150) for inspection.
It does NOT overwrite the original users_new.csv -- merge manually once
you've reviewed the output.
"""
import csv
import os
import numpy as np

RANDOM_STATE = 42
N_MISSING = 50  # U001 .. U050

FIRST_NAMES_M = ['Akash', 'Amit', 'Ankit', 'Arjun', 'Deepak', 'Gaurav', 'Karan',
                  'Nikhil', 'Pankaj', 'Rahul', 'Rajesh', 'Rohit', 'Sanjay',
                  'Suresh', 'Tushar', 'Vikas', 'Vivek']
FIRST_NAMES_F = ['Aisha', 'Ananya', 'Anjali', 'Bhavna', 'Divya', 'Kavya', 'Komal',
                  'Meera', 'Neha', 'Nisha', 'Pooja', 'Priya', 'Riya', 'Shreya',
                  'Simran', 'Sneha', 'Swati', 'Tanya']
LAST_NAMES = ['Gupta', 'Iyer', 'Joshi', 'Kumar', 'Mehta', 'Nair', 'Patel', 'Rao',
              'Reddy', 'Shah', 'Sharma', 'Singh', 'Verma']

LOCATIONS = ['Bangalore', 'Jaipur', 'Ahmedabad', 'Delhi', 'Surat', 'Chennai',
             'Pune', 'Kolkata', 'Hyderabad', 'Mumbai']
LOCATION_WEIGHTS = [0.18, 0.17, 0.11, 0.09, 0.09, 0.08, 0.08, 0.08, 0.07, 0.05]

CLUSTERS = ['health_conscious', 'tea_coffee_fan', 'home_chef', 'snack_lover',
            'busy_professional', 'family_shopper']
CLUSTER_WEIGHTS = [0.27, 0.21, 0.14, 0.14, 0.13, 0.11]

# Deterministic lookup, copied exactly from the existing U051-U150 rows --
# every user in a cluster has this same preferred_categories string.
CLUSTER_TO_CATEGORIES = {
    'home_chef':          'Spices,Grains,Condiments,Dairy',
    'tea_coffee_fan':     'Beverages,Bakery,Snacks,Dairy',
    'health_conscious':   'Health,Grains,Beverages,Dairy',
    'family_shopper':     'Bakery,Dairy,Home Care,Health',
    'busy_professional':  'Noodles,Frozen,Drinks,Snacks',
    'snack_lover':        'Snacks,Bakery,Drinks,Frozen',
}


def find_output_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in ['data/users_new_completed.csv',
                      os.path.join(here, 'users_new_completed.csv')]:
        return candidate  # first candidate is fine; just resolves cwd vs script dir
    

def main():
    rng = np.random.RandomState(RANDOM_STATE)
    rows = []
    for i in range(1, N_MISSING + 1):
        user_id = f"U{i:03d}"
        is_male = rng.random() < 0.52
        gender = 'M' if is_male else 'F'
        first = rng.choice(FIRST_NAMES_M if is_male else FIRST_NAMES_F)
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        age = int(np.clip(round(rng.normal(38.58, 12.11)), 18, 60))
        location = rng.choice(LOCATIONS, p=LOCATION_WEIGHTS)
        cluster = rng.choice(CLUSTERS, p=CLUSTER_WEIGHTS)
        preferred_categories = CLUSTER_TO_CATEGORIES[cluster]
        rows.append({
            'user_id': user_id, 'name': name, 'age': age, 'gender': gender,
            'location': location, 'preferred_categories': preferred_categories,
            'cluster': cluster,
        })

    out_path = 'data/users_new_completed.csv' if os.path.isdir('data') else 'users_new_completed.csv'
    # Merge with existing U051-U150 if present, so the output is the full 150.
    existing_path = 'data/users_new.csv' if os.path.exists('data/users_new.csv') else 'users_new.csv'
    all_rows = list(rows)
    if os.path.exists(existing_path):
        with open(existing_path, encoding='utf-8-sig') as f:
            all_rows.extend(list(csv.DictReader(f)))

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['user_id', 'name', 'age', 'gender',
                                                 'location', 'preferred_categories', 'cluster'])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows ({len(rows)} newly generated U001-U050 "
          f"+ {len(all_rows) - len(rows)} existing) to {out_path}")


if __name__ == '__main__':
    main()
