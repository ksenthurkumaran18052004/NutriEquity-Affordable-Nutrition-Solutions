import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess datasets
nutrition_data = pd.read_csv('Nutrition_Data.csv')
organized_nutritional_needs_data = pd.read_csv('Organized_Nutritional_Needs_Data.csv')

# Preprocessing functions
def convert_age_range(age):
    if '-' in str(age):
        start, end = map(int, age.split('-'))
        return (start + end) / 2
    elif 'and up' in str(age):
        return int(age.split()[0])
    return float(age)

organized_nutritional_needs_data['Age'] = organized_nutritional_needs_data['Age'].apply(convert_age_range)
organized_nutritional_needs_data['Gender'] = organized_nutritional_needs_data['Gender'].str.lower()
organized_nutritional_needs_data['Lifestyle'] = organized_nutritional_needs_data['Lifestyle'].str.lower()
nutrition_data['Diet'] = nutrition_data['Diet'].str.lower()

label_encoders = {}
for column in ['Gender', 'Lifestyle']:
    le = LabelEncoder()
    organized_nutritional_needs_data[column] = le.fit_transform(organized_nutritional_needs_data[column])
    label_encoders[column] = le

# Prepare the model
X = organized_nutritional_needs_data[['Age', 'Gender', 'Lifestyle']].astype('float32')
y = organized_nutritional_needs_data['Calories'].astype('float32')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(input_shape):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = create_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=50, validation_split=0.1, verbose=1)

# Budget Constraints
budget_per_100g = {
    'very poor': 2.0, 'poor': 5.0, 'lower middle class': 10.0,
    'middle class': 15.0, 'upper middle class': 20.0, 'rich': 30.0
}

def find_unique_item(available_items, calories_needed, selected_items):
    if not available_items.empty:
        available_items = available_items.loc[~available_items['name'].isin(selected_items)]
        available_items = available_items.assign(calorie_diff=abs(available_items['calories'] - calories_needed))
        if not available_items.empty:
            recommended_item = available_items.sort_values('calorie_diff').iloc[0]
            return (recommended_item['name'], recommended_item['price ($) per 100g'],
                    recommended_item.get('Brand Name', 'N/A'), recommended_item['calories'])
    return "No meal available", 0.0, "N/A", 0


def recommend_meal(calories_needed, income_level, diet_preference, selected_items):
    max_price = budget_per_100g.get(income_level, 15.0)
    diet_preference = diet_preference.lower()

    available_items = nutrition_data[(nutrition_data['Diet'] == diet_preference) &
                                     (nutrition_data['price ($) per 100g'] <= max_price)]

    calorie_range = 0.5
    for _ in range(3):
        min_calories = calories_needed * (1 - calorie_range)
        max_calories = calories_needed * (1 + calorie_range)
        items_in_calorie_range = available_items[(available_items['calories'] >= min_calories) &
                                                 (available_items['calories'] <= max_calories)]
        
        if not items_in_calorie_range.empty:
            return find_unique_item(items_in_calorie_range, calories_needed, selected_items)
        calorie_range += 0.25

    return find_unique_item(available_items, calories_needed, selected_items)

# Generate multiple meal options with unique items across choice sets
def generate_meal_options(calories_needed, income, diet, num_choices=3):
    meal_choices = []
    used_items = set()
    for _ in range(num_choices):
        selected_items = list(used_items)  # Ensure unique items across sets
        breakfast_item = recommend_meal(calories_needed * 0.3, income, diet, selected_items)
        used_items.add(breakfast_item[0])
        selected_items.append(breakfast_item[0])
        lunch_item = recommend_meal(calories_needed * 0.35, income, diet, selected_items)
        used_items.add(lunch_item[0])
        selected_items.append(lunch_item[0])
        dinner_item = recommend_meal(calories_needed * 0.35, income, diet, selected_items)
        used_items.add(dinner_item[0])
        
        meal_choices.append({
            'breakfast': breakfast_item,
            'lunch': lunch_item,
            'dinner': dinner_item
        })
    return meal_choices

# User input function
def get_user_input():
    age = int(input("Enter your age: "))
    gender = input("Enter your gender (male/female): ").lower()
    income = input("Enter your income level (very poor, poor, lower middle class, middle class, upper middle class, rich): ")
    diet_preference = input("Enter your diet preference (veg/non-veg): ").lower()
    lifestyle = input("Enter your lifestyle condition (Sedentary, Moderately Active, Active): ").lower()
    return age, gender, income, diet_preference, lifestyle

def display_meal_plan(meal_choices):
    print("\nMeal Recommendations (3 choices per meal):\n")
    for idx, choice in enumerate(meal_choices, 1):
        print(f"Choice Set {idx}:")
        breakfast, lunch, dinner = choice['breakfast'], choice['lunch'], choice['dinner']
        print(f"  Breakfast: {breakfast[0]} by {breakfast[2]} - {breakfast[3]:.2f}g at ${breakfast[1]:.2f} per 100g - Total Cost: ${breakfast[1]:.2f}")
        print(f"  Lunch: {lunch[0]} by {lunch[2]} - {lunch[3]:.2f}g at ${lunch[1]:.2f} per 100g - Total Cost: ${lunch[1]:.2f}")
        print(f"  Dinner: {dinner[0]} by {dinner[2]} - {dinner[3]:.2f}g at ${dinner[1]:.2f} per 100g - Total Cost: ${dinner[1]:.2f}\n")
        total_cost = breakfast[1] + lunch[1] + dinner[1]
        print(f"  Total Daily Cost for Choice Set {idx}: ${total_cost:.2f}\n")

def generate_meal_plan():
    age, gender, income, diet, lifestyle = get_user_input()
    input_data = np.array([[age, label_encoders['Gender'].transform([gender])[0],
                            label_encoders['Lifestyle'].transform([lifestyle])[0]]]).astype('float32')
    predicted_calories = model.predict(input_data)[0][0]

    meal_choices = generate_meal_options(predicted_calories, income, diet)
    display_meal_plan(meal_choices)

# Main execution
generate_meal_plan()
