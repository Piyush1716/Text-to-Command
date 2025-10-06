import pandas as pd
import time
from google import genai

# Initialize Gemini client
client = genai.Client(api_key="X")  # or use os.getenv("GEMINI_API_KEY")

# Load your commands dataset
df = pd.read_csv("c.csv")  # columns: command, description
df = df.head()  # test only first rows

all_rows = []

total_commands = len(df)
print(f"Total commands to process: {total_commands}\n")

for idx, row in df.iterrows():
    command = row["command"]
    description = row["description"]

    # Print progress
    print(f"Processing {idx+1}/{total_commands}: {command}")

    prompt = f"""
Generate 12 different natural-language ways a Linux user might ask
to run the command: {command}.
Description: {description}
Return only the queries, one per line.
"""

    try:
        # Call Gemini model
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # or gemini-1.5-flash if not available
            contents=prompt,
        )

        text = response.text

        # Extract queries, clean up
        queries = [q.strip('-•0123456789. ') for q in text.split("\n") if q.strip()]

        # Keep only unique queries
        unique_queries = list(dict.fromkeys(queries))

        # Save each query
        for q in unique_queries[:12]:  # max 12 per command
            all_rows.append({
                "user_query": q,
                "command": command,
                "description": description
            })

    except Exception as e:
        print(f"Error for command {command}: {e}")
        time.sleep(2)

# Save the generated dataset
new_df = pd.DataFrame(all_rows)
new_df.to_csv("natural_queries_gemini.csv", index=False)
print("\nDataset saved to natural_queries_gemini.csv")
