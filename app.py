import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model and scaler
try:
    model = pickle.load(open("model.sav", "rb"))
    df_l = pd.read_csv('first_telc.csv')
    print("Model and dataset loaded successfully")
except Exception as e:
    print(f"Error loading model or dataset: {e}")
    model = None
    df_l = None

def calculate_service_score(metrics):
    """Calculate a service utilization score"""
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    service_count = sum(1 for service in services if metrics.get(service) == 'Yes')
    return service_count / len(services)

def calculate_confidence(probability, metrics):
    """Calculate confidence score with comprehensive feature consideration"""
    # Initialize base confidence (50-80% range)
    base_confidence = 50 + (abs(probability - 0.5) * 60)
    
    # Detailed tenure scoring
    tenure_months = metrics['tenure']
    if tenure_months < 6:
        tenure_score = 0.6 + (tenure_months * 0.05)  # 60-90% for 0-6 months
    elif tenure_months < 12:
        tenure_score = 0.9 + ((tenure_months - 6) * 0.02)  # 90-102% for 6-12 months
    elif tenure_months < 24:
        tenure_score = 1.02 + ((tenure_months - 12) * 0.01)  # 102-114% for 1-2 years
    else:
        tenure_score = 1.14 + (min(tenure_months - 24, 48) * 0.005)  # Max 138% for 4+ years

    # Contract type impact
    contract_scores = {
        'Month-to-month': 0.85,
        'One year': 1.20,
        'Two year': 1.40
    }
    contract_score = contract_scores.get(metrics['contract_type'], 0.85)

    # Service utilization scoring
    services = ['internet_service', 'online_security', 'OnlineBackup', 
                'DeviceProtection', 'tech_support', 'StreamingTV', 'StreamingMovies']
    
    service_count = sum(1 for service in services if metrics.get(service) == 'Yes')
    service_score = 0.9 + (service_count * 0.05)  # 90-125% based on services

    # Payment reliability scoring
    payment_scores = {
        'Electronic check': 0.95,
        'Mailed check': 1.0,
        'Bank transfer': 1.15,
        'Credit card': 1.20
    }
    payment_score = payment_scores.get(metrics.get('PaymentMethod'), 0.95)

    # Charges impact
    monthly_charges = metrics['monthly_charges']
    if monthly_charges < 50:
        charges_score = 1.10
    elif monthly_charges < 80:
        charges_score = 1.05
    elif monthly_charges < 100:
        charges_score = 1.0
    elif monthly_charges < 120:
        charges_score = 0.95
    else:
        charges_score = 0.90

    # Calculate weighted confidence score
    confidence = base_confidence * (
        0.30 * tenure_score +      # 30% weight for tenure
        0.25 * contract_score +    # 25% weight for contract
        0.15 * service_score +     # 15% weight for services
        0.15 * payment_score +     # 15% weight for payment
        0.15 * charges_score       # 15% weight for charges
    )

    # Apply final adjustments and bounds
    if metrics['tenure'] > 36 and contract_score > 1.2:
        confidence *= 1.1  # 10% bonus for loyal customers with long contracts
    
    if service_count >= 5 and payment_score >= 1.15:
        confidence *= 1.05  # 5% bonus for high service usage and reliable payment

    # Ensure reasonable bounds
    confidence = min(95.0, max(30.0, confidence))
    
    # Round to one decimal place
    return round(confidence, 1)

def get_risk_level(probability, metrics):
    """Determine risk level based on probability and customer metrics"""
    base_risk = probability
    
    # Enhanced risk adjustments
    adjustments = {
        'contract_type': {
            'Month-to-month': 0.15,
            'One year': -0.20,
            'Two year': -0.30
        },
        'tenure': {
            'low': 0.12 if metrics['tenure'] < 12 else 0,
            'medium': -0.10 if 12 <= metrics['tenure'] < 36 else 0,
            'high': -0.20 if metrics['tenure'] >= 36 else 0
        },
        'charges': {
            'high': 0.08 if metrics['monthly_charges'] > 100 else 0,
            'very_high': 0.15 if metrics['monthly_charges'] > 150 else 0
        },
        'services': {
            'low': 0.10 if calculate_service_score(metrics) < 0.3 else 0,
            'high': -0.10 if calculate_service_score(metrics) > 0.7 else 0
        }
    }
    
    # Calculate total risk adjustment
    risk_adjustment = (
        adjustments['contract_type'].get(metrics['contract_type'], 0) +
        adjustments['tenure']['low'] +
        adjustments['tenure']['medium'] +
        adjustments['tenure']['high'] +
        adjustments['charges']['high'] +
        adjustments['charges']['very_high'] +
        adjustments['services']['low'] +
        adjustments['services']['high']
    )
    
    adjusted_risk = min(1.0, max(0.0, base_risk + risk_adjustment))
    
    # Enhanced risk level determination
    if adjusted_risk > 0.75:
        return "Very High Risk - Immediate Action Required"
    elif adjusted_risk > 0.60:
        return "High Risk of Customer leaving the services"
    elif adjusted_risk > 0.40:
        return "Moderate Risk of customer leaving the services"
    elif adjusted_risk > 0.25:
        return "Low Risk of Customer to leave the services"
    else:
        return "Very Low Risk of Churn(customer will stay)"


def generate_recommendations(metrics, probability):
    """Generate detailed recommendations based on customer profile"""
    recommendations = []
    
    # High-risk recommendations
    if probability > 0.5:
        if metrics['contract_type'] == 'Month-to-month':
            recommendations.append("Offer discounted long-term contract options")
        
        if metrics['monthly_charges'] > 100:
            recommendations.append("Review current service package for cost optimization")
        
        if metrics.get('tech_support') == 'No':
            recommendations.append("Suggest technical support service enrollment")
        
        if metrics['tenure'] < 12:
            recommendations.append("Implement early engagement program")
            
    # Medium-risk recommendations
    elif probability > 0.3:
        if metrics.get('online_security') == 'No':
            recommendations.append("Promote security service benefits")
        
        if metrics.get('device_protection') == 'No':
            recommendations.append("Highlight device protection advantages")
    
    # General recommendations
    service_score = calculate_service_score(metrics)
    if service_score < 0.5:
        recommendations.append("Present bundle package options")
    
    return recommendations

@app.route('/')
def loadpage():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or df_l is None:
        return jsonify({'error': 'Model or dataset not loaded properly'})
    
    try:
        # Get form data
        form_data = request.get_json()
        
        # Create customer metrics dictionary
        customer_metrics = {
            'monthly_charges': float(form_data['query1']),
            'total_charges': float(form_data['query2']),
            'tenure': float(form_data['query3']),
            'contract_type': form_data['query13'],
            'tech_support': form_data['query10'],
            'online_security': form_data['query7'],
            'device_protection': form_data['query9'],
            'internet_service': form_data['query6']
        }
        
        # Create input DataFrame
        input_data = {
            'MonthlyCharges': customer_metrics['monthly_charges'],
            'TotalCharges': customer_metrics['total_charges'],
            'tenure': customer_metrics['tenure'],
            'Partner': form_data['query4'],
            'Dependents': form_data['query5'],
            'InternetService': form_data['query6'],
            'OnlineSecurity': form_data['query7'],
            'OnlineBackup': form_data['query8'],
            'DeviceProtection': form_data['query9'],
            'TechSupport': form_data['query10'],
            'StreamingTV': form_data['query11'],
            'StreamingMovies': form_data['query12'],
            'Contract': form_data['query13'],
            'PaperlessBilling': form_data['query14'],
            'PaymentMethod': form_data['query15']
        }
        
        new_df = pd.DataFrame([input_data])
        
        # Combine with existing data
        df_2 = pd.concat([df_l, new_df], ignore_index=True)
        
        # Create tenure groups
        labels = ["1-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
        df_2['tenure_group'] = pd.cut(df_2['tenure'].astype(float), 
                                    bins=[0, 12, 24, 36, 48, 60, 72], 
                                    labels=labels,
                                    include_lowest=True)
        df_2 = df_2.drop(columns=['tenure'])
        
        # Create dummy variables
        categorical_cols = df_2.select_dtypes(include=['object']).columns
        df_dummies = pd.get_dummies(df_2[categorical_cols])
        
        # Add numeric columns
        numeric_cols = ['MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            df_dummies[col] = df_2[col]
        
        # Ensure all model features are present
        for col in model.feature_names_in_:
            if col not in df_dummies.columns:
                df_dummies[col] = 0
                
        # Get final input for prediction
        final_input = df_dummies[model.feature_names_in_].tail(1)
        
        # Make prediction
        probabilities = model.predict_proba(final_input)[0]
        churn_probability = probabilities[1]
        retention_probability = probabilities[0]
        
        # Generate insights
        risk_level = get_risk_level(churn_probability, customer_metrics)
        confidence = calculate_confidence(churn_probability, customer_metrics)
        recommendations = generate_recommendations(customer_metrics, churn_probability)
        
        # Generate impact factors
        impact_factors = []
        
        if customer_metrics['contract_type'] == 'Month-to-month':
            impact_factors.append("Short-term contract increases volatility")
        
        if customer_metrics['monthly_charges'] > 100:
            impact_factors.append(f"Higher than average monthly charges (${customer_metrics['monthly_charges']:.2f})")
        
        if customer_metrics['tenure'] < 12:
            impact_factors.append(f"New customer relationship ({int(customer_metrics['tenure'])} months)")
        elif customer_metrics['tenure'] > 36:
            impact_factors.append(f"Long-term customer ({int(customer_metrics['tenure'])} months)")
        
        service_score = calculate_service_score(customer_metrics)
        if service_score < 0.5:
            impact_factors.append("Limited service utilization")
        
        # Prepare response
        result = {
            'prediction': risk_level,
            'probabilities': {
                'churn': f'{churn_probability * 100:.1f}%',
                'retention': f'{retention_probability * 100:.1f}%'
            },
            'confidence': f'{confidence:.1f}%',
            'customer_metrics': {
                'monthly_charges': customer_metrics['monthly_charges'],
                'total_charges': customer_metrics['total_charges'],
                'contract_type': customer_metrics['contract_type'],
                'tenure': customer_metrics['tenure']
            },
            'analysis': {
                'primary_factors': impact_factors,
                'specific_recommendations': recommendations,
                'recommendation': (
                    "Immediate attention required" if churn_probability > 0.6
                    else "Monitor closely" if churn_probability > 0.4
                    else "Regular monitoring"
                )
            }
        }
        
        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Error in prediction process',
            'details': str(e),
            'trace': traceback.format_exc()
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
