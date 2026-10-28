"""export_weather_json.py : exporte le dernier rapport de weather.db en JSON (sans images, ≈150 Ko).

Le fichier produit peut être déposé dans « Importer des rapports » du tableau de bord générique (app.py).

Usage :  python export_weather_json.py [weather.db] [sortie.json] [report_id]
"""
import json
import sys
from pathlib import Path

from weather_db import WeatherDB

db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "weather.db"
if not db_path.exists():
    sys.exit(f"Base introuvable : {db_path}")

db = WeatherDB(db_path, read_only=True)
rid = sys.argv[3] if len(sys.argv) > 3 else db.latest_report_id()
if rid is None:
    sys.exit("weather.db ne contient aucun rapport.")

out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"{rid}.json")
report = db.get_report(rid, include_images=False)
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Rapport {rid} exporté -> {out} ({out.stat().st_size / 1024:.0f} Ko)")
