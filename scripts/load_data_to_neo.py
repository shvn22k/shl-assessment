import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv, dotenv_values
from tqdm import tqdm

env_vars = dotenv_values(".env")
load_dotenv(".env", override=True)

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "assessments_raw.json")

uri = env_vars.get("NEO4J_URI") or os.getenv("NEO4J_URI")
user = env_vars.get("NEO4J_USER") or os.getenv("NEO4J_USER")
password = env_vars.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(user, password))

def create_constraints(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Assessment) REQUIRE n.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:TestType) REQUIRE t.name IS UNIQUE")

def upsert_node(tx, node):
    tx.run(
        """
        MERGE (a:Assessment {name: $name})
        SET a.remote_testing = $remote_testing,
            a.adaptive_irt = $adaptive_irt,
            a.link = $link,
            a.description = $description,
            a.duration = $duration,
            a.job_levels = $job_levels
        """,
        name=node["name"],
        remote_testing=node.get("remote_support", "No"),
        adaptive_irt=node.get("adaptive_support", "No"),
        link=node.get("url", ""),
        description=node.get("description", ""),
        duration=node.get("duration", ""),
        job_levels=node.get("job_levels", [])
    )

def create_relationships(tx, node):
    test_types = node.get("test_types", [])
    if isinstance(test_types, list):
        for test_type in test_types:
            tx.run(
                """
                MERGE (t:TestType {name: $test_type})
                MERGE (a:Assessment {name: $name})
                MERGE (a)-[:IS_OF_TYPE]->(t)
                """,
                name=node["name"],
                test_type=test_type
            )

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    with driver.session() as session:
        session.execute_write(create_constraints)
        for node in tqdm(data, desc="Uploading Assessments"):
            session.execute_write(upsert_node, node)
            session.execute_write(create_relationships, node)

    driver.close()
    print("Done loading assessments into Neo4j!")

if __name__ == "__main__":
    main()
