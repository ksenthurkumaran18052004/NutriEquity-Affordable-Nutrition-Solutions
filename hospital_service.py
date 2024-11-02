import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# Load hospital data
file_path = "hospital_dataset.csv" 
hospital_data = pd.read_csv(file_path)

# Initialize geolocator
geolocator = Nominatim(user_agent="hospital_locator")

def get_coordinates(zip_code):
    """Fetch coordinates for a given ZIP code."""
    try:
        location = geolocator.geocode(zip_code)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"Error fetching coordinates for ZIP {zip_code}: {e}")
    return None

def calculate_distance(zip1, zip2):
    """Calculate the distance between two ZIP codes in miles."""
    if zip1 == zip2:
        return 0  # If ZIP codes are the same, return distance as zero
    coord1, coord2 = get_coordinates(zip1), get_coordinates(zip2)
    if coord1 and coord2:
        return round(geodesic(coord1, coord2).miles, 2)
    return None

# List of available medical problems/services
medical_services = [
    "MRI", "X-Ray", "Blood Test", "Appendectomy", "CT Scan",
    "Physical Exam", "Knee Replacement", "Cardiac Stress Test",
    "Colonoscopy", "Ultrasound"
]

# Display available services
print("Available medical problems/services:")
for idx, service in enumerate(medical_services, start=1):
    print(f"{idx}. {service}")

# Taking user inputs from terminal
selected_service_index = int(input("\nEnter the number corresponding to the medical service you need: ")) - 1
user_medical_problem = medical_services[selected_service_index]
user_zip = input("Enter your ZIP code: ")
max_price = int(input("Enter your maximum price range: "))
max_wait_time = int(input("Enter your maximum wait time (in minutes): "))

# Filter hospitals based on user criteria
filtered_df = hospital_data[
    (hospital_data["Service Name"].str.lower() == user_medical_problem.lower()) &
    (hospital_data["Price (USD)"] <= max_price) &
    (hospital_data["Average Wait Time (min)"] <= max_wait_time)
].copy()

# Calculate distances
filtered_df["Distance (miles)"] = filtered_df["ZIP"].apply(lambda z: calculate_distance(user_zip, str(z)))

# Drop entries with missing distances
filtered_df = filtered_df.dropna(subset=["Distance (miles)"])

# Sort the results by distance
filtered_df = filtered_df.sort_values(by="Distance (miles)")

# Display final filtered result
result = filtered_df[["Hospital Name", "Address", "Service Name", "Price (USD)", 
                      "Average Wait Time (min)", "Distance (miles)", "Hospital Rating"]]

print("\nRecommended Hospitals based on your criteria:")
print(result)
