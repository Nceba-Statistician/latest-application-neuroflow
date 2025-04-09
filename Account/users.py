# If you're worried about overwriting files with the same name, you can do:

import uuid, os
uploaded_CSV_file = ""
save_path = ""
unique_suffix = uuid.uuid4().hex[:6]
name, ext = os.path.splitext(uploaded_CSV_file)
# name, ext = os.path.splitext(uploaded_CSV_file.name)
new_file_name = f"{name}_{unique_suffix}{ext}"
full_path = os.path.join(save_path, new_file_name)

