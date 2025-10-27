# import sqlite3
# import numpy as np

# DB_PATH = r"C:\Face csec\Face csec\Facelive\database\student.db"
# conn = sqlite3.connect(DB_PATH)
# cursor = conn.cursor()

# # Assume you have 'embedding_id' and 'student_id' in your students table
# cursor.execute("SELECT student_id FROM students ORDER BY embedding_id ASC")
# rows = cursor.fetchall()

# student_ids = np.array([row[0] for row in rows])
# np.save(r"Facelive/database/embedding_id_to_student.npy", student_ids)
# print("Mapping file created:", student_ids)
import sqlite3
import numpy as np
import os

# Ensure database folder exists
os.makedirs("database", exist_ok=True)

DB_PATH = r"database/student.db"
SAVE_PATH = r"database/embedding_id_to_student.npy"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Fetch student IDs ordered by embedding_id
cursor.execute("SELECT student_id FROM students ORDER BY embedding_id ASC")
rows = cursor.fetchall()

student_ids = np.array([row[0] for row in rows])
np.save(SAVE_PATH, student_ids)

print("Mapping file created:", student_ids)
