# app.py (FINALIZED for Email-Login & ReportLab)

from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import select
from database import Session, UserCredentials, HealthRecord, initialize_database 
from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd

# --- Visualization Imports ---
import matplotlib
matplotlib.use('Agg') # Server side rendering ke liye zaroori
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# --- ReportLab Imports (WeasyPrint ki jagah) ---
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# --- ML Models Load Karna (R3.1) ---
MODEL_DIABETES = None
MODEL_HEART = None
try:
    with open('model_diabetes.pkl', 'rb') as f:
        MODEL_DIABETES = pickle.load(f)
    with open('model_heart.pkl', 'rb') as f:
        MODEL_HEART = pickle.load(f)
    print("ML Models successfully loaded.")
except FileNotFoundError:
    print("WARNING: ML model files (model_*.pkl) not found. Run ml_models.py first.")

# --- Flask App Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_for_session' 
initialize_database() 


# --- Helper Function for Visualization (R5.1, R5.2) ---

def plot_health_trends(df, metric, time_filter='weekly'):
    """Diye gaye DataFrame ke mutabiq health trend ka graph banata hai."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if time_filter == 'daily': period = timedelta(days=1)
    elif time_filter == 'monthly': period = timedelta(days=30)
    else: period = timedelta(days=7)

    start_date = datetime.utcnow() - period
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_filtered = df[df['timestamp'] >= start_date].sort_values('timestamp')
    
    if df_filtered.empty or len(df_filtered) < 2:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, 'Not enough data available for trend analysis.', ha='center', va='center', fontsize=12); ax.axis('off'); return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_filtered['timestamp'], df_filtered[metric], marker='o', linestyle='-', color='teal')
    
    ax.set_title(f'{metric.replace("_", " ").title()} Trend ({time_filter.capitalize()})', fontsize=14)
    ax.set_xlabel('Date and Time'); ax.set_ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='png'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# --- Helper Function for AI Prediction (R3.2, R3.4, R4.4) ---

def get_risk_analysis(record_data):
    """
    R3.2: User ke latest record par prediction models chalata hai.
    R3.4: Risk level (Low/Medium/High) assign karta hai.
    """
    
    if not MODEL_DIABETES or not MODEL_HEART:
        return {'risk': 'High', 'message': 'AI Models are not loaded. Cannot calculate risk.', 'details': [], 'recommendations': []}

    risk_level = "Low"
    details = []
    recommendations = ["Keep recording your health data regularly.", "Maintain a balanced diet and hydration."]

    if record_data.systolic_bp > 140 or record_data.diastolic_bp > 90:
        risk_level = "High"; details.append("Alert: High Blood Pressure detected."); recommendations.append("Consult a doctor immediately regarding your BP readings.")
    elif record_data.systolic_bp > 130 or record_data.diastolic_bp > 85:
        if risk_level != "High": risk_level = "Medium"; details.append("Warning: Elevated Blood Pressure (Hypertension Risk)."); recommendations.append("Reduce salt intake and monitor BP daily.")
    
    if record_data.sugar_level > 125:
        if risk_level == "Low": risk_level = "Medium"; details.append("Warning: High Sugar Level detected (Diabetes Risk)."); recommendations.append("Focus on a low sugar diet and increase fiber intake.")
    
    if record_data.stress_level >= 8 and record_data.sleep_hours < 6:
         if risk_level == "Low": risk_level = "Medium"; details.append("Warning: High Stress and low sleep hours detected."); recommendations.append("Practice stress management techniques and ensure 7-8 hours of sleep.")
    
    try:
        diabetes_input = np.array([1, record_data.sugar_level, record_data.systolic_bp, 30, 100, record_data.weight * 0.4, 0.4, 30]).reshape(1, -1)
        prob_diabetes = MODEL_DIABETES.predict_proba(diabetes_input)[0][1]
        
        if prob_diabetes > 0.45:
            if risk_level == "Low": risk_level = "Medium"; details.append(f"AI Prediction: Diabetes Risk Probability is {prob_diabetes*100:.1f}%."); recommendations.append("Increase your daily exercise to improve insulin sensitivity.")

    except Exception as e:
        details.append(f"AI Prediction failed for Diabetes.")

    return {
        'risk': risk_level, 
        'message': f"Overall Risk Level: {risk_level}", 
        'details': details,
        'recommendations': list(set(recommendations))
    }


# --- Routes (Pages) ---

@app.route('/')
def index():
    """Default route: Agar user logged in hai toh dashboard, warna login."""
    if 'email' in session: # Session check 'email' par shift
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Email-based Login functionality."""
    if request.method == 'POST':
        email = request.form.get('email') # Form se email lena
        password = request.form.get('password')
        
        db_session = Session()
        # Query ko 'email' par shift karna
        user = db_session.scalar(select(UserCredentials).where(UserCredentials.email == email))
        db_session.close()

        if user and check_password_hash(user.password_hash, password):
            session['email'] = user.email # Session mein email store karna
            session['username'] = user.username # Session mein username store karna
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check your email and password.', 'danger')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Email-based Signup functionality."""
    if request.method == 'POST':
        # New fields: email aur username
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        
        hashed_password = generate_password_hash(password)
        
        db_session = Session()
        # Check if email already exists
        if db_session.scalar(select(UserCredentials).where(UserCredentials.email == email)):
            flash('Email already exists!', 'danger')
            db_session.close()
            return redirect(url_for('signup'))

        # Data save karte waqt email aur username dono store karna
        new_user = UserCredentials(email=email, username=username, password_hash=hashed_password, role='user')
        db_session.add(new_user)
        db_session.commit()
        db_session.close()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')


@app.route('/dashboard')
def dashboard():
    """R6.1: Main Dashboard (Updated to check for 'email')."""
    if 'email' not in session: # Session check 'email' par shift
        flash('Please login to access the dashboard.', 'warning')
        return redirect(url_for('login'))

    # Session se email aur username uthana
    current_username = session['username']
    
    # Latest record fetch karna (Risk Analysis ke liye)
    db_session = Session()
    # Linking HealthRecord to session['username']
    latest_record = db_session.scalars(
        select(HealthRecord)
        .where(HealthRecord.username == current_username)
        .order_by(HealthRecord.timestamp.desc())
    ).first()
    db_session.close()

    risk_result = None
    if latest_record:
        risk_result = get_risk_analysis(latest_record)

    return render_template('dashboard.html', username=current_username, risk_result=risk_result)

@app.route('/add_record', methods=['GET', 'POST'])
def add_record():
    """Health Data Entry Form dikhata hai aur data save karta hai (Updated to use session['username'])."""
    if 'email' not in session:
        flash('Please login to add health records.', 'warning')
        return redirect(url_for('login'))

    username = session['username']
    
    if request.method == 'POST':
        try:
            # R1.2: Form se data lena aur type casting
            weight = float(request.form.get('weight'))
            systolic_bp = float(request.form.get('systolic_bp'))
            diastolic_bp = float(request.form.get('diastolic_bp'))
            sugar_level = float(request.form.get('sugar_level'))
            sleep_hours = float(request.form.get('sleep_hours'))
            exercise_minutes = float(request.form.get('exercise_minutes'))
            stress_level = int(request.form.get('stress_level'))
            
            # R1.3: Basic input validation
            if weight <= 0 or sleep_hours < 0 or stress_level < 1 or stress_level > 10:
                 flash('Invalid input detected. Please enter realistic values.', 'danger')
                 return render_template('data_entry.html', username=username)

            # R1.4: Database mein record save karna
            db_session = Session()
            new_record = HealthRecord(
                username=username, # Linking via username
                weight=weight,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                sugar_level=sugar_level,
                sleep_hours=sleep_hours,
                exercise_minutes=exercise_minutes,
                stress_level=stress_level,
                timestamp=datetime.utcnow()
            )
            db_session.add(new_record)
            db_session.commit()
            db_session.close()
            
            flash('Health record successfully added!', 'success')
            return redirect(url_for('dashboard'))

        except ValueError:
            flash('Invalid format! Please ensure all inputs are numbers.', 'danger')
        except Exception as e:
            flash(f'An error occurred: {e}', 'danger')

    return render_template('data_entry.html', username=username)

# --- R1.5: View, Edit, Delete Routes ---

@app.route('/view_records')
def view_records():
    if 'email' not in session:
        flash('Please login to view health records.', 'warning')
        return redirect(url_for('login'))

    db_session = Session()
    records = db_session.scalars(
        select(HealthRecord)
        .where(HealthRecord.username == session['username']) # Linking via username
        .order_by(HealthRecord.timestamp.desc())
    ).all()
    db_session.close()

    return render_template('view_records.html', records=records)

@app.route('/edit_record/<int:record_id>', methods=['GET', 'POST'])
def edit_record(record_id):
    if 'email' not in session:
        flash('Please login to edit records.', 'warning')
        return redirect(url_for('login'))

    db_session = Session()
    record = db_session.get(HealthRecord, record_id)

    if not record or record.username != session['username']:
        db_session.close()
        flash('Record not found or access denied.', 'danger')
        return redirect(url_for('view_records'))

    if request.method == 'POST':
        try:
            record.weight = float(request.form.get('weight'))
            record.systolic_bp = float(request.form.get('systolic_bp'))
            record.diastolic_bp = float(request.form.get('diastolic_bp'))
            record.sugar_level = float(request.form.get('sugar_level'))
            record.sleep_hours = float(request.form.get('sleep_hours'))
            record.exercise_minutes = float(request.form.get('exercise_minutes'))
            record.stress_level = int(request.form.get('stress_level'))
            
            db_session.commit()
            flash('Health record successfully updated!', 'success')
            return redirect(url_for('view_records'))

        except ValueError:
            flash('Invalid format! Please ensure all inputs are numbers.', 'danger')
        except Exception as e:
            flash(f'An error occurred during update: {e}', 'danger')
        finally:
            db_session.close()

    db_session.close()
    return render_template('edit_record.html', record=record)

@app.route('/delete_record/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    if 'email' not in session:
        flash('Please login to delete records.', 'warning')
        return redirect(url_for('login'))
    
    db_session = Session()
    record = db_session.get(HealthRecord, record_id)

    if not record or record.username != session['username']:
        db_session.close()
        flash('Record not found or access denied.', 'danger')
        return redirect(url_for('view_records'))

    try:
        db_session.delete(record)
        db_session.commit()
        flash('Health record successfully deleted.', 'info')
    except Exception as e:
        flash(f'An error occurred during deletion: {e}', 'danger')
    finally:
        db_session.close()
    
    return redirect(url_for('view_records'))

# --- R5: Trends Visualization Route ---

@app.route('/trends')
def trends():
    """R5: Health Visualization & Trends Module."""
    if 'email' not in session:
        flash('Please login to view trends.', 'warning')
        return redirect(url_for('login'))

    time_filter = request.args.get('filter', 'weekly')
    
    db_session = Session()
    records_obj = db_session.scalars(
        select(HealthRecord)
        .where(HealthRecord.username == session['username']) # Linking via username
        .order_by(HealthRecord.timestamp.desc())
    ).all()
    db_session.close()

    if not records_obj:
        flash("Aapke paas abhi koi data nahi hai trends dekhne ke liye. Please record add karein.", 'info')
        return redirect(url_for('dashboard'))

    # Records ko Pandas DataFrame mein convert karna
    df = pd.DataFrame([r.__dict__ for r in records_obj])
    df = df.drop(columns=['_sa_instance_state'], errors='ignore')
    
    # Graphs generate karna
    try:
        weight_plot = plot_health_trends(df, 'weight', time_filter)
        bp_plot = plot_health_trends(df, 'systolic_bp', time_filter)
        sugar_plot = plot_health_trends(df, 'sugar_level', time_filter)
    except Exception as e:
        flash(f"Visualization mein masla: {e}", 'danger')
        weight_plot, bp_plot, sugar_plot = None, None, None

    return render_template('trends.html', 
                           weight_plot=weight_plot,
                           bp_plot=bp_plot,
                           sugar_plot=sugar_plot,
                           current_filter=time_filter)

# --- R5.4: PDF Report Generation Route (ReportLab) ---

@app.route('/generate_report')
def generate_report():
    """
    R5.4: User ke records aur analysis ko lekar PDF report generate karta hai (using ReportLab).
    """
    if 'email' not in session:
        flash('Please login to generate a report.', 'warning')
        return redirect(url_for('login'))

    username = session['username']
    
    db_session = Session()
    records_obj = db_session.scalars(
        select(HealthRecord)
        .where(HealthRecord.username == session['username'])
        .order_by(HealthRecord.timestamp.desc())
    ).all()
    
    latest_record = records_obj[0] if records_obj else None
    db_session.close()

    if not latest_record:
        flash("Record nahi mila. Pehle data add karein.", 'danger')
        return redirect(url_for('dashboard'))

    risk_result = get_risk_analysis(latest_record)
    
    # 1. ReportLab Setup
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Risk Styles
    high_risk_style = ParagraphStyle(
        name='HighRisk', fontSize=12, leading=15, backColor=colors.HexColor('#f8d7da'), borderColor=colors.red, borderWidth=1, borderPadding=5, borderRadius=3, alignment=0
    )
    medium_risk_style = ParagraphStyle(
        name='MediumRisk', fontSize=12, leading=15, backColor=colors.HexColor('#fff3cd'), borderColor=colors.orange, borderWidth=1, borderPadding=5, borderRadius=3, alignment=0
    )
    low_risk_style = ParagraphStyle(
        name='LowRisk', fontSize=12, leading=15, backColor=colors.HexColor('#d4edda'), borderColor=colors.green, borderWidth=1, borderPadding=5, borderRadius=3, alignment=0
    )
    
    Story = []

    # 2. Report Header
    Story.append(Paragraph(f"<b>Smart Health Monitoring Report for {username}</b>", styles['Title']))
    Story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    Story.append(Spacer(1, 0.5 * 72))

    # 3. Risk Analysis Section (R3)
    Story.append(Paragraph("<b>1. Latest Risk Analysis & Alerts</b>", styles['h2']))
    
    # Select appropriate style based on risk level
    if risk_result['risk'] == 'High':
        risk_style = high_risk_style
    elif risk_result['risk'] == 'Medium':
        risk_style = medium_risk_style
    else:
        risk_style = low_risk_style
        
    Story.append(Paragraph(f"<b>{risk_result['message']}</b>", risk_style))
    
    # Details and Recommendations
    Story.append(Paragraph("<br/><b>Risk Details:</b>", styles['Normal']))
    for detail in risk_result['details']:
        Story.append(Paragraph(f"&bull; {detail}", styles['Normal']))

    Story.append(Paragraph("<br/><b>Recommendations:</b>", styles['Normal']))
    for rec in risk_result['recommendations']:
        Story.append(Paragraph(f"&bull; {rec}", styles['Normal']))

    Story.append(Spacer(1, 0.5 * 72))

    # 4. Detailed Records Table (R1)
    Story.append(Paragraph("<b>2. Detailed Health Records (Latest Entries)</b>", styles['h2']))
    
    if records_obj:
        data = [['Date', 'Weight (kg)', 'BP (Sys/Dia)', 'Sugar (mg/dL)', 'Sleep (hrs)', 'Stress (1-10)']]
        for r in records_obj:
            data.append([
                r.timestamp.strftime('%Y-%m-%d %H:%M'),
                str(r.weight),
                f"{r.systolic_bp}/{r.diastolic_bp}",
                str(r.sugar_level),
                str(r.sleep_hours),
                str(r.stress_level),
            ])
            
        table = Table(data, colWidths=[1.5*72, 1*72, 1.2*72, 1*72, 1*72, 1*72])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.25, colors.black),
            ('FONTSIZE', (0,0), (-1,-1), 9),
        ]))
        Story.append(table)
    else:
        Story.append(Paragraph("No detailed records found.", styles['Normal']))


    # 5. Build the PDF
    doc.build(Story)
    pdf_file = buffer.getvalue(); buffer.close()
    response = make_response(pdf_file)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={username}_Health_Report.pdf'
    
    flash('PDF report successfully generated!', 'success')
    return response

@app.route('/logout')
def logout():
    """Logout functionality."""
    session.pop('email', None) # Logout mein email aur username dono pop kar rahe hain
    session.pop('username', None) 
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Flask app ko run karna
    app.run(debug=True)