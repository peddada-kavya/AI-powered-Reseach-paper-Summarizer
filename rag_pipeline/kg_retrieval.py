from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "neo4j123"   # change if different


class KG:

    def __init__(self):

        self.driver = GraphDatabase.driver(
            URI,
            auth=(USER, PASSWORD)
        )


    def retrieve(self, user_query):

        cypher = """
        MATCH (a)-[r]->(b)
        WHERE toLower(a.text) CONTAINS toLower($search_query)
           OR toLower(b.text) CONTAINS toLower($search_query)
        RETURN a.text AS source,
               type(r) AS relation,
               b.text AS target
        LIMIT 5
        """

        with self.driver.session() as session:

            # ✅ FIXED LINE (dictionary parameter)
            results = session.run(
                cypher,
                {"search_query": user_query}
            )

            context = []

            for record in results:

                source = record["source"]
                relation = record["relation"]
                target = record["target"]

                context.append(
                    f"{source} {relation} {target}"
                )

            return context


if __name__ == "__main__":

    kg = KG()

    print(kg.retrieve("Artificial Intelligence"))
