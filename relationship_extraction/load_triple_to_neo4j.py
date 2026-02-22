from neo4j import GraphDatabase
import json

# Neo4j connection details
uri = "neo4j://127.0.0.1:7687"
username = "neo4j"
password = "123neo4j"  # your password

driver = GraphDatabase.driver(uri, auth=(username, password))

# Load triples
with open("triples.json", "r", encoding="utf-8") as f:
    triples = json.load(f)

def create_graph(tx, subject, predicate, object_):
    tx.run(
        """
        MERGE (p:Paper {name: $subject})
        MERGE (e:Entity {name: $object})
        MERGE (p)-[:RELATION {type: $predicate}]->(e)
        """,
        subject=subject,
        predicate=predicate,
        object=object_
    )

# Write to Neo4j
with driver.session() as session:
    for t in triples:
        session.execute_write(
            create_graph,
            t["subject"],
            t["predicate"],
            t["object"]
        )

driver.close()
print("✅ Triples successfully loaded into Neo4j")
