import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tasks import ingest_file_task

class DocFolderHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)

    def process_file(self, file_path):
        fname = os.path.basename(file_path)
        ext = os.path.splitext(fname)[1].lower()
        # Filter for document types
        if ext in ('.txt', '.pdf', '.md', '.docx', '.csv', '.html', '.json'):
            print(f"[WATCHDOG] Detected change in: {fname}. Queueing ingestion...")
            # Trigger Celery background task under public user
            ingest_file_task.delay(file_path, fname, "public")

if __name__ == "__main__":
    docs_dir = "./docs"
    os.makedirs(docs_dir, exist_ok=True)
    
    event_handler = DocFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, docs_dir, recursive=False)
    observer.start()
    print(f"[WATCHDOG] Monitoring directory '{docs_dir}' for file changes...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
