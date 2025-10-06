import pandas as pd
import time
import os
from google import genai

# Initialize Gemini client
client = genai.Client(api_key='X')

# Load your commands dataset
df = pd.read_csv("c.csv")  # columns: command, description
# df = df.head(30)  # uncomment to test on small subset
# df = df.iloc[30:60,:]  # uncomment to test on small subset
# df = df.iloc[60:100,:]  # uncomment to test on small subset
# df = df.iloc[300:350,:]  # uncomment to test on small subset
df = df.iloc[450:500,:]  # uncomment to test on small subset

batch_size = 10
output_file = "gem_450_500.csv"

# If resuming, load existing results to avoid duplication
if os.path.exists(output_file):
    all_rows = pd.read_csv(output_file).to_dict("records")
    processed = {row["command"] for row in all_rows}
    print(f"Resuming... already processed {len(processed)} commands.")
else:
    all_rows = []
    processed = set()

total_commands = len(df)
print(f"Total commands to process: {total_commands}\n")

for start in range(0, total_commands, batch_size):
    batch = df.iloc[start:start + batch_size]

    for idx, row in batch.iterrows():
        command = row["command"]
        description = row["description"]

        if command in processed:
            print(f"Skipping already processed: {command}")
            continue

        print(f"Processing {idx+1}/{total_commands}: {command}")

        prompt = f"""
Generate 12 different natural-language ways a Linux user might ask
to run the command: {command}.
Description: {description}
Return only the queries, one per line.
"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            text = response.text

            # Extract queries
            queries = [q.strip("-•0123456789. ") for q in text.split("\n") if q.strip()]
            unique_queries = list(dict.fromkeys(queries))

            for q in unique_queries[:12]:
                all_rows.append({
                    "user_query": q,
                    "command": command,
                    "description": description
                })
            processed.add(command)

        except Exception as e:
            print(f"Error for command {command}: {e}")
            time.sleep(5)  # wait before retry/next
            continue

    # Save progress after each batch
    pd.DataFrame(all_rows).to_csv(output_file, index=False)
    print(f"✅ Saved progress after batch {start//batch_size + 1}")

print("\nAll done! Final dataset saved to:", output_file)
