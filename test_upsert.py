from supabase import create_client

SUPABASE_URL = "https://buybylvazbbtznchupjw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ1eWJ5bHZhemJidHpuY2h1cGp3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIwNjgwMTksImV4cCI6MjA2NzY0NDAxOX0.HIVWcIwkgCzqslqw9UymzjGi73-6PAJaQSb1_OEdeQE"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

test_data = [{
    "last_search_event": "test_event_001",
    "outlet_name": "4401:5601 Pizza Vizza",
    "created_at": "2025-07-09T10:00:00",
    "city_code": "std:01",
    "buyer_app": "test_app",
    "account_name": "test_account",
    "message": "test message",
    "status": "active",
    "provider_id": "4401:5601",
    "outlet_name_clean": "Pizza Vizza",
    "date": "2025-07-09"
}]

try:
    response = supabase.table("search_data").upsert(
        test_data,
        on_conflict=['last_search_event', 'outlet_name']
    ).execute()
    print("Upsert test succeeded:", response)
except Exception as e:
    print("Upsert test failed:", e)
