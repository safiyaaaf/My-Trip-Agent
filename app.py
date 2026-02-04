import os, json, base64, urllib.parse, zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field
from openai import OpenAI

# --------------------------------
# Setup
# --------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

UAE_TZ = ZoneInfo("Asia/Dubai")
AMS_TZ = ZoneInfo("Europe/Amsterdam")

DEFAULT_UNSAFE_AREAS = """Amsterdam Zuidoost (Bijlmer / Bijlmermeer / Bullewijk / Amstel III)
Nieuw-West: Slotervaart (Deflandplein), Osdorp, Geuzenveld–Slotermeer, Wildemanbuurt
Centrum: Oude Zijde / Red Light District streets late at night
Amsterdam Centraal / Damrak (pickpockets / scams corridor)
"""

# --------------------------------
# Data models
# --------------------------------
class ExtractedItem(BaseModel):
    type: str
    title: str
    date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    location_text: Optional[str] = None
    notes: Optional[str] = None

class ExtractionResult(BaseModel):
    items: List[ExtractedItem] = []
    raw_text_summary: Optional[str] = None

class PlanStop(BaseModel):
    name: str
    location_text: str
    start_time: str
    end_time: str
    activity_type: str
    notes: Optional[str] = None

class DayPlan(BaseModel):
    plan_name: str
    local_timezone: str = "Europe/Amsterdam"
    stops: List[PlanStop]
    assumptions: List[str] = []

# --------------------------------
# Helpers
# --------------------------------
def img_to_data_url(uploaded_file) -> str:
    b = uploaded_file.getvalue()
    b64 = base64.b64encode(b).decode("utf-8")
    mime = uploaded_file.type or "image/jpeg"
    return f"data:{mime};base64,{b64}"

def uae_to_amsterdam_time(date_str: str, time_str: str) -> str:
    dt_uae = datetime.fromisoformat(f"{date_str}T{time_str}:00").replace(tzinfo=UAE_TZ)
    dt_ams = dt_uae.astimezone(AMS_TZ)
    return dt_ams.strftime("%H:%M")

def maps_dir_link(origin: str, destination: str, mode: str) -> str:
    params = {"api": 1, "origin": origin, "destination": destination, "travelmode": mode}
    return "https://www.google.com/maps/dir/?" + urllib.parse.urlencode(params)

def safe_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        repair = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": [{"type": "input_text", "text": "Fix to valid JSON only:\n" + text}]}],
        )
        return json.loads(repair.output_text)

def parse_kml_or_kmz(file_bytes: bytes, filename: str):
    if filename.lower().endswith(".kmz"):
        z = zipfile.ZipFile(BytesIO(file_bytes))
        kml_name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
        if not kml_name:
            return []
        kml_data = z.read(kml_name)
    else:
        kml_data = file_bytes

    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    root = ET.fromstring(kml_data)

    places = []
    for pm in root.findall(".//kml:Placemark", ns):
        name_el = pm.find("kml:name", ns)
        name = name_el.text.strip() if name_el is not None and name_el.text else "Unnamed place"

        coord_el = pm.find(".//kml:Point/kml:coordinates", ns)
        coords = coord_el.text.strip() if coord_el is not None and coord_el.text else None

        lat = lon = None
        if coords:
            parts = coords.split(",")
            if len(parts) >= 2:
                lon, lat = parts[0].strip(), parts[1].strip()

        places.append({"name": name, "lat": lat, "lon": lon})

    return places

# --------------------------------
# AI functions
# --------------------------------
def extract_from_images(image_urls: List[str]) -> ExtractionResult:
    prompt = """
Return ONLY valid JSON:

{
  "items": [{
    "type": "flight|hotel|event|train|museum|restaurant|other",
    "title": "",
    "date": "YYYY-MM-DD or null",
    "start_time": "HH:MM or null",
    "end_time": "HH:MM or null",
    "location_text": "",
    "notes": ""
  }],
  "raw_text_summary": ""
}
"""
    content = [{"type": "input_text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "input_image", "image_url": url})

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": content}],
    )
    return ExtractionResult(**safe_json(resp.output_text))

def plan_day(extraction, mymaps, date_str, home, wake_uae, late_by, sleep_late, avoid, transport):
    wake_ams = uae_to_amsterdam_time(date_str, wake_uae)
    mymaps_txt = "\n".join([f"- {p['name']}" for p in mymaps[:200]]) if mymaps else "None"

    prompt = f"""
Create TWO plans (JSON ONLY):

Inputs:
Date: {date_str}
Home: {home}
Wake UAE: {wake_uae}
Wake AMS: {wake_ams}
Late by minutes: {late_by}
Sleep late: {sleep_late}

Avoid areas:
{avoid}

My Maps places:
{mymaps_txt}

Extracted:
{extraction.model_dump_json()}

Return schema:
{{
 "planA": {{ "plan_name":"Plan A","local_timezone":"Europe/Amsterdam","stops":[],"assumptions":[] }},
 "planB_90min_late": {{ "plan_name":"Plan B","local_timezone":"Europe/Amsterdam","stops":[],"assumptions":[] }}
}}
"""
    resp = client.responses.create(
        model="gpt-4.1",
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )
    return safe_json(resp.output_text), wake_ams

# --------------------------------
# UI
# --------------------------------
st.set_page_config(page_title="My Trip Agent", layout="wide")
st.title("My Trip Agent — Amsterdam")
st.caption("Screenshots + Google My Maps → smart daily plan")
run_btn = st.button("✅ Create my plan", type="primary")
st.divider()

date = st.text_input("Date (YYYY-MM-DD)")
home = st.text_input("Home / Hotel")
wake = st.text_input("Wake time (UAE HH:MM)", "09:00")
sleep_late = st.checkbox("I prefer slow mornings", True)
late_by = st.slider("Running late (minutes)", 0, 180, 0, 10)

mymaps_file = st.file_uploader("Upload My Maps (KML/KMZ)", type=["kml", "kmz"])
mymaps_places = parse_kml_or_kmz(mymaps_file.getvalue(), mymaps_file.name) if mymaps_file else []

avoid = st.text_area("Areas to avoid", DEFAULT_UNSAFE_AREAS)
screenshots = st.file_uploader("Upload screenshots", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

# -----------------------------
# Run planner
# -----------------------------
if run_btn:
    if not (date and home and screenshots):
        st.error("Please fill the date, home/hotel, and upload at least one screenshot.")
    else:
        with st.spinner("Creating your smart plan..."):
            urls = [img_to_data_url(f) for f in screenshots]

            extraction = extract_from_images(urls)

            plans, wake_ams = plan_day(
                extraction,
                mymaps_places,
                date,
                home,
                wake,
                late_by,
                sleep_late,
                avoid,
                "transit"
            )

            st.success(f"Wake time in Amsterdam: {wake_ams}")
            st.json(plans)
