import backend.DB as DB
import json
from backend.queries import query_list
from backend.queries import insert_operation
import time

def get_collection_count():
    return DB.eu.count()

def get_collection_stats():
    stats = DB.db.command("collstats", "eu")
    return {k: stats[k] for k in ('count', 'nindexes', 'size')} 

def insert_json(json_obj):
    start = time.process_time()
    json_decode = json.loads(json_obj)
    inserted = insert_operation(json_decode)
    time_elapsed = time.process_time() - start
    return (inserted, time_elapsed)

def performance_evaluation():
    start_time = time.time()
    print("Started performance evaluation", flush=True)
    print("", flush=True)
    for num, fn in enumerate(query_list):
        with open(".query.state", 'w+') as file:
            percent = (num / len(query_list)) * 100 
            file.write(f"{str(percent)}: Running {fn.__name__}")
        query_start = time.time()
        try:
            fn()
        except:
            with open(".query.state", 'w+') as file:
                file.write(f"100:Error - Query {fn.__name__} failed")
            return
        print(f"Finished iteration {num+1} of {len(query_list)} (time elapsed {(time.time() - query_start):.1f}s) func: {fn.__name__} ", flush=True)
    print("Finished performance evaluation", flush=True)
    print("", flush=True)

    time_elapsed = (time.time() - start_time)
    with open(".query.state", 'w+') as file:
        file.write(f"100:Done - Time elapsed {time_elapsed:.3f} seconds")
