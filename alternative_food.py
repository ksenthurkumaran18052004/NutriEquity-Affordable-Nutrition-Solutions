import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from fuzzywuzzy import fuzz, process

# Load the data
file_path = 'Nutrition_Data_Price.csv'
nutrition_data = pd.read_csv(file_path)

# Select features related to nutritional values
nutritional_features = [
    'calories', 'total_fat', 'saturated_fat', 'cholesterol', 
    'sod m', 'choline', 'folate', 'vitamin_a', 'vitamin_c', 
    'calcium', 'iron', 'protein', 'carbohydrate', 'fiber'
]

# Filter the dataset to only include relevant nutritional and price features
data_features = nutrition_data[nutritional_features + ['name', 'price ($) per 100g']]
food_names = data_features['name'].str.lower().tolist()  # Convert to lowercase for matching

# Scale the nutritional data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_features[nutritional_features])

# Fit KNN model
knn_model = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn_model.fit(scaled_features)

# Function to summarize nutrition
def summarize_nutrition(nutritional_data):
    summary = []
    if nutritional_data['protein'] > 8:
        summary.append("High in protein")
    if nutritional_data['sod m'] > 300:
        summary.append("High in sodium")
    if nutritional_data['fiber'] > 5:
        summary.append("High in fiber")
    if nutritional_data['iron'] > 2:
        summary.append("High in iron")
    if nutritional_data['calcium'] > 100:
        summary.append("High in calcium")
    if nutritional_data['vitamin_c'] > 10:
        summary.append("High in vitamin C")
    if nutritional_data['vitamin_a'] > 50:
        summary.append("High in vitamin A")
    if nutritional_data['total_fat'] > 15:
        summary.append("High in fat")
    if nutritional_data['saturated_fat'] > 5:
        summary.append("High in saturated fat")
    if nutritional_data['carbohydrate'] > 10:
        summary.append("High in carbohydrates")
    return ', '.join(summary)

def find_alternative(food_name, max_price):
    # Convert food name to lowercase for consistency
    food_name = food_name.lower()
    
    # Find all food names containing the input term
    matching_items = [name for name in food_names if food_name in name]
    
    # If no matches, inform the user
    if not matching_items:
        print(f"No items found containing '{food_name}' in the dataset.")
        return
    
    # If multiple matches, prompt the user to choose one
    if len(matching_items) > 1:
        print(f"Multiple items found containing '{food_name}':")
        for i, item in enumerate(matching_items, 1):
            print(f"{i}. {item.title()}")
        try:
            selection = int(input("Enter the number of your choice (or 0 to exit): "))
            if selection == 0:
                print("Exiting without finding alternatives.")
                return
            elif 1 <= selection <= len(matching_items):
                best_match = matching_items[selection - 1]
            else:
                print("Invalid choice. Exiting.")
                return
        except ValueError:
            print("Invalid input. Exiting.")
            return
    else:
        best_match = matching_items[0]
    
    # Find the index of the matched food
    food_index = data_features[data_features['name'].str.lower() == best_match].index[0]
    food_data = scaled_features[food_index].reshape(1, -1)
    
    # Get the nutritional values and price of the input food
    input_food_info = data_features.loc[food_index, nutritional_features + ['price ($) per 100g']]
    print(f"\nSelected Food: {best_match.title()}")
    print(f"Price: ${input_food_info['price ($) per 100g']:.2f} per 100g")
    print("Nutritional Summary:", summarize_nutrition(input_food_info))
    
    # Find the 10 nearest neighbors
    distances, indices = knn_model.kneighbors(food_data)
    
    # Get alternatives that are cheaper and close in nutritional value
    print("\nCheaper Alternatives:")
    original_price = data_features.loc[food_index, 'price ($) per 100g']
    
    found_alternative = False
    for idx, dist in zip(indices[0][1:], distances[0][1:]):  # Skipping the first item as it is the same food
        alternative = data_features.iloc[idx]
        
        # Only consider items that are substantially cheaper and nutritionally similar
        if alternative['price ($) per 100g'] < max_price and alternative['price ($) per 100g'] < original_price * 0.8:
            print(f"\nAlternative Food: {alternative['name']}")
            print(f"Price: ${alternative['price ($) per 100g']:.2f} per 100g")
            print("Nutritional Summary:", summarize_nutrition(alternative))
            found_alternative = True
    
    if not found_alternative:
        print(f"No cheaper alternatives found for '{best_match.title()}' within the given price range.")

# Get user input
food_name = input("Enter the food item name: ")
try:
    max_price = float(input("Enter the maximum price for alternatives: "))
    find_alternative(food_name, max_price)
except ValueError:
    print("Please enter a valid number for the price.")
