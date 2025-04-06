"""
Mock data generator for fire brigade incident reports.
Creates realistic incident reports with timestamps.
"""
import random
import datetime
import uuid

# Sample incidents with sentiment categories
POSITIVE_REPORTS = [
    "Successfully rescued family from burning building with no injuries.",
    "Quick response led to complete fire containment with minimal damage.",
    "Team effectively managed chemical spill with zero environmental impact.",
    "Saved three cats from a tree without incident.",
    "Rescued elderly resident from flood waters, receiving community praise.",
    "Fire extinguished before spreading to neighboring properties.",
    "Training exercise completed with excellent team coordination.",
    "All personnel returned safely after dangerous rescue operation.",
    "New equipment performed exceptionally well during emergency response.",
    "Community expressed gratitude for rapid response to gas leak.",
    "Successfully evacuated school during fire drill with record completion time.",
    "Rescue team saved trapped hikers with precision helicopter extraction.",
    "New water pump system demonstrated superior performance during high-rise fire.",
    "Volunteer team received commendation for flood relief efforts.",
    "Fire prevention program successfully implemented in local schools.",
    "Coordinated effort with police led to successful evacuation of mall.",
    "Emergency medical training resulted in successful on-site resuscitation.",
    "Wildfire contained before reaching residential areas thanks to rapid response.",
    "Rescue boat team successfully retrieved stranded kayakers from river.",
    "New communication protocol significantly improved response coordination.",
    "Hazmat team contained industrial leak with no civilian exposure.",
    "Drone deployment enabled successful rescue of hiker in remote area.",
    "CPR training for community members resulted in saved life at local event.",
    "Fire department charity event raised record funds for burn victims.",
    "Quick-thinking firefighter prevented gas explosion at apartment complex.",
    "Thermal imaging cameras enabled successful location of child in smoke-filled room.",
    "Cross-department training improved regional emergency response capabilities.",
    "Team managed to save historic building structure despite extensive fire.",
    "Rescue of trapped construction workers completed without injuries.",
    "Department received excellence award for community safety initiatives."
]

NEUTRAL_REPORTS = [
    "Responded to fire alarm which was determined to be a false alarm.",
    "Conducted routine equipment inspection and maintenance.",
    "Attended monthly safety briefing with all staff.",
    "Dispatched to reported smoke, found controlled barbecue.",
    "Inspected fire safety systems at local business.",
    "Provided standby support at community event.",
    "Meeting held with council regarding disaster planning.",
    "Training session conducted on new protocols.",
    "Quarterly drill completed as scheduled.",
    "Routine patrol of high-risk area conducted.",
    "Updated emergency contact information for all personnel.",
    "Completed standard vehicle maintenance on ladder truck.",
    "Monitored controlled burn at local farm property.",
    "Performed hydrant flow testing in eastern district.",
    "Conducted inventory of medical supplies at station.",
    "Filed monthly incident reports with headquarters.",
    "Attended cross-departmental coordination meeting.",
    "Replaced batteries in station smoke detectors.",
    "Standard shift change completed with no outstanding issues.",
    "Updated maps of district emergency water sources.",
    "Completed CPR recertification for five team members.",
    "Observed smoke was from authorized industrial venting.",
    "Responded to automatic alarm; system reset required.",
    "Documented training hours for department certification.",
    "Conducted routine test of backup generator system.",
    "Staff attended county emergency management briefing.",
    "Standard rotation of stored fuel completed.",
    "Received updated weather notification system.",
    "Performed weekly radio check with all stations.",
    "Routine inspection of personal protective equipment completed."
]

NEGATIVE_REPORTS = [
    "Multiple injuries reported due to building collapse during firefighting operation.",
    "Equipment failure hampered rescue attempts at apartment fire.",
    "Response time delayed due to traffic congestion, increasing property damage.",
    "Volunteer firefighter suffered smoke inhalation requiring hospitalization.",
    "Flood response limited by insufficient resources and personnel.",
    "Communication system failed during critical emergency coordination.",
    "Vehicle collision en route to emergency caused additional injuries.",
    "Budget constraints prevented purchase of essential safety equipment.",
    "High winds spread fire beyond containment lines, destroying additional structures.",
    "Staff shortage led to inadequate coverage during major incident.",
    "Structural collapse trapped two firefighters during warehouse fire.",
    "Breathing apparatus malfunction forced evacuation of rescue team.",
    "Extreme heat conditions caused heat exhaustion in multiple firefighters.",
    "Water supply failure severely impacted firefighting capabilities.",
    "Hazardous material exposure sent three responders to hospital.",
    "Radio dead zones prevented coordination in multi-story building.",
    "Outdated maps led team to blocked access route, delaying response.",
    "Electrical failure disabled critical pumping equipment during major fire.",
    "Civilian injuries increased due to premature reentry to evacuated building.",
    "Rescue helicopter grounded due to mechanical issues during critical operation.",
    "Inadequate training for industrial fire led to ineffective initial response.",
    "Emergency vehicle involved in collision while responding to call.",
    "Backdraft injured two firefighters during commercial building fire.",
    "Ladder failure during rescue operation resulted in firefighter injury.",
    "Water contamination from fire runoff created environmental hazard.",
    "Power line contact electrocuted responder during storm cleanup.",
    "Inadequate protective gear led to chemical exposure injuries.",
    "Mutual aid response delayed due to jurisdictional confusion.",
    "Faulty hydrant connection caused significant water pressure loss.",
    "Dispatch error sent team to wrong location, delaying emergency response."
]

# Add variations to reports by adding modifiers
MODIFIERS = {
    "time": ["Yesterday", "Last night", "This morning", "At 3 AM", "During evening shift", 
             "During day shift", "Over the weekend", "Last Thursday", "Earlier today", "Last hour"],
    "location": ["in the downtown area", "on Main Street", "at the industrial park", "in the suburban district", 
                 "near the shopping mall", "at the river crossing", "in the north district", "at the city center", 
                 "near the highway", "in the apartment complex"],
    "result": ["resulting in commendation from the chief", "leading to community recognition", "which was documented for training purposes", 
               "requiring additional resources", "causing service disruption", "requiring extensive cleanup", 
               "prompting policy review", "necessitating equipment inspection", "triggering protocol updates", "with minimal disruption"]
}

def generate_variant(report):
    """Generate a variant of a report by adding modifiers."""
    # 50% chance to add a time modifier at the beginning
    if random.random() < 0.5:
        report = random.choice(MODIFIERS["time"]) + ", " + report[0].lower() + report[1:]
    
    # 30% chance to add a location
    if random.random() < 0.3:
        # If the report ends with a period, insert the location before it
        if report.endswith('.'):
            report = report[:-1] + " " + random.choice(MODIFIERS["location"]) + "."
        else:
            report = report + " " + random.choice(MODIFIERS["location"]) + "."
    
    # 20% chance to add a result
    if random.random() < 0.2:
        # If the report ends with a period, replace it with a comma and add the result
        if report.endswith('.'):
            report = report[:-1] + ", " + random.choice(MODIFIERS["result"]) + "."
        else:
            report = report + ", " + random.choice(MODIFIERS["result"]) + "."
            
    return report

def generate_random_incident():
    """
    Generate a random incident report with timestamp and ID.
    
    Returns:
        dict: A dictionary containing incident details
    """
    # Randomly select sentiment category
    sentiment_category = random.choice(["positive", "neutral", "negative"])
    
    # Select report based on sentiment
    if sentiment_category == "positive":
        base_report = random.choice(POSITIVE_REPORTS)
    elif sentiment_category == "neutral":
        base_report = random.choice(NEUTRAL_REPORTS)
    else:
        base_report = random.choice(NEGATIVE_REPORTS)
    
    # Generate a variant of the report
    report = generate_variant(base_report)
    
    # Generate random timestamp within the last 30 days
    days_ago = random.randint(0, 30)
    hours_ago = random.randint(0, 23)
    minutes_ago = random.randint(0, 59)
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=days_ago, hours=hours_ago, minutes=minutes_ago)
    
    # Format timestamp as ISO string
    formatted_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Generate unique ID
    incident_id = str(uuid.uuid4())[:8]
    
    return {
        "incident_id": incident_id,
        "report": report,
        "timestamp": formatted_timestamp,
        "_true_sentiment": sentiment_category  # Hidden field for validation
    }

def generate_incidents(count=10):
    """
    Generate multiple random incidents.
    
    Args:
        count (int): Number of incidents to generate
        
    Returns:
        list: List of incident dictionaries
    """
    return [generate_random_incident() for _ in range(count)]

def generate_specific_incident(sentiment_category):
    """
    Generate an incident with a specific sentiment category.
    
    Args:
        sentiment_category (str): "positive", "neutral", or "negative"
        
    Returns:
        dict: Incident dictionary
    """
    # Select report based on sentiment
    if sentiment_category == "positive":
        base_report = random.choice(POSITIVE_REPORTS)
    elif sentiment_category == "neutral":
        base_report = random.choice(NEUTRAL_REPORTS)
    else:
        base_report = random.choice(NEGATIVE_REPORTS)
    
    # Generate a variant of the report
    report = generate_variant(base_report)
    
    # Generate random timestamp within the last 30 days
    days_ago = random.randint(0, 30)
    hours_ago = random.randint(0, 23)
    minutes_ago = random.randint(0, 59)
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=days_ago, hours=hours_ago, minutes=minutes_ago)
    
    # Format timestamp as ISO string
    formatted_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Generate unique ID
    incident_id = str(uuid.uuid4())[:8]
    
    return {
        "incident_id": incident_id,
        "report": report,
        "timestamp": formatted_timestamp,
        "_true_sentiment": sentiment_category
    }

if __name__ == "__main__":
    # Test the generator
    print(generate_incidents(3)) 