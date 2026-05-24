"""
make_dataset.py — Generate a synthetic NHS patient feedback dataset.

This script creates two XLS-format files in data/raw/ that mirror the
column structure of the real NHS Choices dataset (data.gov.uk, OGL licence).

Use this if you don't have the original XLS files, or to reproduce a clean
development dataset with a fixed random seed.

Usage:
    python data/make_dataset.py

Outputs:
    data/raw/hospital_comments.xls   (600 rows)
    data/raw/gp_comments.xls         (400 rows)

To use the real NHS Choices data instead:
    1. Download from https://www.data.gov.uk/dataset/73740ffe-cecb-4cba-afb9-51ea996187a1
    2. Rename:
         "Comments regarding hospitals and responses.xls" -> data/raw/hospital_comments.xls
         "Comments regarding GPs and responses.xls"       -> data/raw/gp_comments.xls
    3. Re-run train_model.py
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import openpyxl

random.seed(42)

# ---------------------------------------------------------------------------
# Comment pools — one per service-improvement theme
# ---------------------------------------------------------------------------

HOSPITAL_COMMENTS = [
    # appointment_access
    "I tried to book an appointment for three weeks and could never get through on the phone. The booking system is completely broken.",
    "Getting an appointment was a nightmare. I called every day for two weeks before I finally got seen.",
    "The online booking system kept crashing. Nobody at reception was able to help me book over the phone either.",
    "I was told there were no available appointments for six weeks. This is unacceptable for a non-emergency but urgent matter.",
    "Reception staff were helpful when trying to book but the system simply did not have any slots available for a month.",
    "Booking was straightforward and easy online. I got an appointment within two days which was excellent.",
    "The new online booking system is fantastic. I got an appointment the next day without even having to call.",
    "I could not get an appointment for weeks. I ended up going to A&E because I could not wait any longer.",
    "The phone lines are busy every single morning. I gave up trying to call and just turned up at the clinic.",
    "Appointment booking via the app worked perfectly. Very impressed with how quick and easy it was.",
    # communication
    "Nobody explained what the procedure would involve. I went home with no idea what to expect or what the next steps were.",
    "The consultant explained everything clearly and made sure I understood. I left feeling informed and reassured.",
    "I was kept waiting for results for three weeks with no update. Nobody contacted me to explain the delay.",
    "The doctor took time to explain my diagnosis in plain language. I really appreciated that.",
    "After my operation nobody told me what aftercare I needed. I had to find out myself online.",
    "Communication from the ward was excellent. Nurses kept me updated throughout my stay.",
    "I was told I would get a letter within two weeks but it never arrived. When I called, nobody could find my file.",
    "The team were very good at explaining things in a way I could understand. No jargon at all.",
    "I received conflicting information from different members of staff. This caused me a great deal of anxiety.",
    "The pre-operative appointment was thorough and left me with no questions unanswered.",
    # staff_attitude
    "The receptionist was dismissive and rude when I asked a simple question. I felt like I was an inconvenience.",
    "Every member of staff I encountered was warm, kind and professional. Outstanding care.",
    "The nurse who treated me was absolutely wonderful. She was patient and explained everything clearly.",
    "I was made to feel stupid for asking questions. The doctor barely made eye contact during the consultation.",
    "Staff were rushed off their feet but still took time to be kind and caring. Very impressive.",
    "The consultant was arrogant and dismissive. I did not feel listened to at all.",
    "All the nursing staff were incredibly compassionate and went above and beyond.",
    "I was treated with dignity and respect throughout my entire visit. Thank you to all the staff.",
    "The doctor seemed irritated by my questions and cut me off several times. Very poor manner.",
    "The entire team made a frightening experience feel manageable. I cannot thank them enough.",
    # waiting_time
    "I waited four hours in A&E before anyone saw me. The waiting area was overcrowded and there were no seats.",
    "My outpatient appointment was delayed by two hours with no explanation given.",
    "Despite arriving on time for my appointment, I waited over ninety minutes in the waiting room.",
    "The wait was longer than expected but staff kept us informed of delays which was appreciated.",
    "I was in and out within twenty minutes. The most efficient hospital visit I have ever had.",
    "I waited six hours in A&E for what turned out to be a minor issue. The wait was exhausting.",
    "My appointment started on time and the whole process was quick and efficient.",
    "The queue for the pharmacy after discharge was over an hour long. This needs to be addressed.",
    "Waiting times were clearly displayed on the screens in the waiting area. Very helpful and transparent.",
    "I understand the NHS is under pressure but a three hour wait for a scheduled appointment is too long.",
    # treatment_quality
    "The treatment I received was first class. My condition was diagnosed quickly and the procedure was done with great skill.",
    "I am not confident the correct diagnosis was made. My symptoms continued for months after I was told everything was fine.",
    "The surgical team were outstanding. My recovery has been excellent and I am very grateful.",
    "I was prescribed medication that I had previously told them caused me side effects. No one checked my records.",
    "The physiotherapy treatment was excellent. My mobility has improved significantly in just six weeks.",
    "The pain management after my procedure was inadequate. I was in significant discomfort overnight.",
    "My diagnosis was explained clearly and the treatment plan was thorough and well considered.",
    "I was sent home too quickly after surgery and had to be readmitted the following day.",
    "The care I received during my stay was exceptional. I felt safe and well looked after throughout.",
    "The doctor seemed unsure about my diagnosis and kept changing their mind. This was very worrying.",
    # administration
    "My referral letter was lost and I had to start the referral process again from scratch. This delayed my treatment by two months.",
    "My test results were not sent to my GP. I had to chase both the hospital and my practice repeatedly.",
    "The discharge letter was clear, detailed and sent to my GP promptly. Very impressed.",
    "I waited six weeks for a letter confirming my follow up appointment which still had the wrong date on it.",
    "The prescription was ready on time and the process was smooth and well organised.",
    "My prescription was not sent to the pharmacy as promised. I had to go back to the surgery to sort it out.",
    "Paperwork was handled efficiently and my records were clearly up to date.",
    "My medical records from my previous hospital had not been transferred even though I requested this months ago.",
    "The admin team were helpful when I called about my referral and resolved my query quickly.",
    "The outpatient letters contain too much medical jargon. A plain English summary would be very helpful.",
    # facilities
    "The ward was clean and well maintained throughout my stay.",
    "The toilets in the outpatient waiting area were dirty and had clearly not been cleaned for some time.",
    "Parking was impossible. I ended up being late for my appointment because I could not find a space.",
    "The equipment in the clinic was clearly outdated. I hope investment is planned for this area.",
    "The hospital is clean, modern and well laid out. Navigation signs are excellent.",
    "The car park charges are extremely expensive. This is a significant burden for patients attending regular appointments.",
    "The waiting room was cramped and uncomfortable, especially for patients with mobility issues.",
    "The new wing is excellent. Bright, clean and much more comfortable than the older parts of the building.",
    "There was no disabled parking available when I arrived for my appointment.",
    "The facilities were good and the ward felt calm and well organised.",
]

GP_COMMENTS = [
    "I call at 8am every morning and the lines are always engaged. By the time I get through all the appointments are gone.",
    "It took me three weeks to get a routine appointment. The access to GP appointments in this area is very poor.",
    "The online booking system is brilliant. I got an appointment within 48 hours without any difficulty.",
    "I was told I could not have an appointment for two weeks unless I called on the day. This system does not work for working people.",
    "The practice has introduced an online triage system which is very efficient. I was contacted within the same day.",
    "Phone lines are constantly busy. I spent forty minutes trying to get through before I gave up.",
    "Getting an appointment has become much easier since they introduced the new booking system.",
    "I had to visit A&E for something that should have been handled by my GP because I could not get an appointment in time.",
    "The receptionist was able to book me in quickly when I explained my symptoms. Very helpful.",
    "Routine appointments are fine but urgent on the day appointments are nearly impossible to get.",
    "My doctor explained my test results clearly and took time to answer all my questions.",
    "I received a text saying my results were ready but when I called nobody could tell me what they showed.",
    "The GP called me back promptly and was very thorough in explaining my options.",
    "I was given very little information about the medication I was prescribed. No leaflet or explanation.",
    "The practice sends regular newsletters and health reminders which are very useful.",
    "After my blood test nobody contacted me for four weeks. It turned out the result required follow up.",
    "The doctor was excellent at explaining my diagnosis in a way I could understand.",
    "I asked for a referral letter and was told it had been sent but my consultant never received it.",
    "The practice nurse called me to follow up after my procedure which I thought was very caring.",
    "Information about the practice opening hours and services is very difficult to find online.",
    "The receptionist questioned why I needed an appointment in a way that felt invasive and inappropriate.",
    "All the staff at this practice are brilliant. I always feel welcome and well looked after.",
    "My GP is outstanding. Patient, knowledgeable and always makes me feel that my concerns are valid.",
    "I felt rushed during my appointment. The doctor seemed to want me out of the door as quickly as possible.",
    "The nurse practitioner was thorough and professional. A very reassuring consultation.",
    "The receptionist was very helpful and kind when I was upset. That made a real difference.",
    "I have been with this practice for years and the quality of care has always been excellent.",
    "The doctor interrupted me several times and seemed disinterested in what I was saying.",
    "Every interaction with this practice has been positive. Wonderful people doing a difficult job.",
    "I did not feel listened to during my consultation. My concerns were brushed aside.",
    "I arrived on time but was kept waiting for forty five minutes past my appointment time.",
    "The practice runs on time and I have never had to wait more than five minutes.",
    "The waiting room was very full and the wait was over an hour for an on the day appointment.",
    "My appointment was on time and the consultation was thorough without feeling rushed.",
    "I have noticed significant delays at this practice over the last six months.",
    "Telephone consultations are much quicker and I am always called within the agreed time window.",
    "I waited twenty minutes past my appointment but the doctor apologised and explained there had been an emergency.",
    "The new triage system means I often get a call back within hours rather than waiting days for an appointment.",
    "My GP referred me immediately and the referral was appropriate and timely.",
    "I feel my condition has been managed poorly. I have been on the same medication for two years with no review.",
    "The doctor carried out a thorough examination and I left confident in the diagnosis and treatment plan.",
    "I have been back three times with the same symptoms and each time given a different explanation.",
    "The annual health check was comprehensive and gave me great peace of mind.",
    "My medication review was rushed and I did not feel the doctor really engaged with my concerns.",
    "The GP spotted a problem that had been overlooked for years. I am very grateful for their thoroughness.",
    "I was prescribed antibiotics without a proper examination. I did not feel this was appropriate.",
    "My repeat prescription was ready on time and the process has always been smooth.",
    "I have been waiting eight weeks for a referral letter that should have been sent within two.",
    "The practice was quick to send my records to my new GP when I moved. Very efficient.",
    "My test results were sent to the wrong department and the whole process had to start again.",
]

HOSPITALS = [
    "Royal Victoria Infirmary", "Manchester Royal Infirmary", "Leeds General Infirmary",
    "St Thomas Hospital", "University College Hospital", "Nottingham City Hospital",
    "Sheffield Teaching Hospital", "Bristol Royal Infirmary", "Newcastle General Hospital",
    "King's College Hospital", "St George's Hospital", "Aintree University Hospital",
]

GP_PRACTICES = [
    "Riverside Medical Centre", "The Elms Surgery", "Parkview Health Centre",
    "Central Medical Practice", "Northside Surgery", "Greenbank Practice",
    "The Family Practice", "Millbrook Medical Centre", "Lakeside Surgery",
    "Valley Road Practice", "Highfields Medical Centre", "Bridge Street Surgery",
]

RESPONSES = [
    "Thank you for your feedback. We are sorry to hear about your experience and will look into this.",
    "We appreciate you taking the time to share your experience. Your comments will be reviewed by the team.",
    "Thank you for your kind words. We will share your feedback with the staff involved.",
    "We are very sorry this was your experience. Please contact our PALS team if you would like to discuss this further.",
    "Thank you for your positive feedback. It is great to hear that our staff made a difference.",
    "", "", "",  # many comments have no response
]

RATINGS = [1, 2, 3, 4, 5]
RATING_WEIGHTS = [0.08, 0.10, 0.15, 0.27, 0.40]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_date(start_year: int = 2008, end_year: int = 2010) -> str:
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 5, 1)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%d/%m/%Y")


def _make_workbook(comments: list, organisations: list, n: int) -> openpyxl.Workbook:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Comments"
    ws.append(["OrganisationName", "Comment", "Title", "Date",
               "Rating", "Response", "ResponseDate"])
    for i in range(1, n + 1):
        comment  = random.choice(comments)
        org      = random.choice(organisations)
        rating   = random.choices(RATINGS, weights=RATING_WEIGHTS)[0]
        response = random.choice(RESPONSES)
        resp_dt  = _random_date() if response else ""
        ws.append([org, comment, f"Patient comment {i}",
                   _random_date(), rating, response, resp_dt])
    return wb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_dir = Path(__file__).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    hosp_path = raw_dir / "hospital_comments.xls"
    gp_path   = raw_dir / "gp_comments.xls"

    print("Generating synthetic NHS patient feedback dataset...")

    hosp_wb = _make_workbook(HOSPITAL_COMMENTS, HOSPITALS, n=600)
    hosp_wb.save(hosp_path)
    print(f"  {hosp_path}  ({os.path.getsize(hosp_path):,} bytes,  600 rows)")

    gp_wb = _make_workbook(GP_COMMENTS, GP_PRACTICES, n=400)
    gp_wb.save(gp_path)
    print(f"  {gp_path}  ({os.path.getsize(gp_path):,} bytes,  400 rows)")

    print("\nDone. Run 'python train_model.py' to train the model.")


if __name__ == "__main__":
    main()
