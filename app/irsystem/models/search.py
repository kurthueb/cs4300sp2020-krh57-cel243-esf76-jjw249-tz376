import math
import json
from collections import defaultdict
import numpy as np
def cosine_similarity(joined_queries, eligible, work_mat, auth_mat, positive_query_works, query_works, positive_query_authors, query_authors, works, authors):
    """Returns a list of work ids ranked by similarity to a query. Does not use formal cosine similarity due to the omission of normalizing by the doc norm. 

    Arguments
    =========

    joined_queries: np matrix,
        A vector representing the joined queries and author preferences. This is what is used to calculate overall similarity

    Returns
    ======
    ranked_results: list of size len(tf_idf_vectors)
        ranked_results[i] = work id of the i'th most relevant work
    """
    results = []

    similarity_scores = np.matmul(work_mat, joined_queries)
    for i in np.argsort(-similarity_scores):
        if i in eligible and i not in query_works:
            results.append((i, similarity_scores[i]))
        if len(results)==100:
            break
    # if len(positive_query_works) == 0 and len(positive_query_authors) == 0:
    #     return results       

    def penalize(val):
        if val>0:
            val = val**0.4
        else:
            val = -((-val)**0.4)
        return val

    reordered_results = []
    for work, score in results:
        cosine_sims = []
        for query in positive_query_works:
            # so having dot products 0.5, 0.5 is better than 0.9, 0.1
            val = np.dot(work_mat[query], work_mat[work])
            cosine_sims.append(penalize(val))
        for query in positive_query_authors:
            val = np.dot(auth_mat[query], work_mat[work])
            cosine_sims.append(penalize(val))
        reordered_results.append((work, np.mean(np.array(cosine_sims))))
    reordered_results.sort(key=lambda x: x[1], reverse=True)

    scores_by_query = {}
    for work, score in reordered_results:
        scores_by_query[work] = []
        for query_id in query_works:
            raw_similarity = np.dot(work_mat[query_id], work_mat[work])
            raw_similarity = math.floor((raw_similarity - (-1))/2*100)
            liked_or_disliked = query_id in positive_query_works
            title = works[query_id]["title"]
            scores_by_query[work].append({"liked": liked_or_disliked, "title": title, "score": raw_similarity})

        for author_id in query_authors:
            raw_similarity = np.dot(auth_mat[author_id], work_mat[work])
            raw_similarity = math.floor((raw_similarity - (-1))/2*100)
            liked_or_disliked = author_id in positive_query_authors
            author = authors[author_id]
            scores_by_query[work].append({"liked": liked_or_disliked, "title": author, "score": raw_similarity})

        scores_by_query[work].sort(key=lambda x: x["score"], reverse=True)
        
    return reordered_results, scores_by_query

def combine_queries(work_ids, auth_ids, work_mat, auth_mat, works):
    """

    Arguments
    =========

    work_ids: list,
        A list of works and their scores in the query
        
    
    Returns
    ======
    query: dict mapping terms to tf-idf values
    """
    dimensions = len(work_mat[0])
    combined_queries = np.zeros(dimensions)
    rocchio_adjustment = np.zeros(dimensions)
    for query_work in work_ids:
        weight = query_work["score"]
        vector_id = query_work["work_id"]
        combined_queries += work_mat[vector_id]*weight

        work_rocchio = np.zeros(dimensions)
        for sim_work in works[query_work["work_id"]]["similar_works"]:
            work_rocchio += work_mat[sim_work]
        rocchio_adjustment += weight * (work_rocchio)

    for query_author in auth_ids:
        weight = query_author["score"]
        vector_id = query_author["auth_id"]
        combined_queries += auth_mat[vector_id]*weight

    combined_queries = combined_queries + (0.7) * rocchio_adjustment
    query_norm = np.linalg.norm(combined_queries)
    if query_norm > 0.0001:
        combined_queries = combined_queries/query_norm

    return combined_queries


def get_doc_rankings(work_ids, eligible, auth_ids, work_mat, auth_mat, works, authors):
    """Returns a dictionary of terms and tf-idf values representing the combined result of individual queries

    Arguments
    =========

    work_ids: list,
        A list of works in the query


    Returns
    ======
    results_list: A JSON-formatted list of dictionaries containing K/V pairs for title, author, ranking, book_url, image_url, and description.
    """
    positive_query_works = []
    positive_query_authors = []
    query_works = []
    query_authors = []
    for query in work_ids:
        if query["score"] > 0:
            positive_query_works.append(query["work_id"])
        query_works.append(query["work_id"])

    for query in auth_ids:
        if query["score"] > 0:
            positive_query_authors.append(query["auth_id"])
        query_authors.append(query["auth_id"])
    joined_queries = combine_queries(work_ids, auth_ids, work_mat, auth_mat, works)
    ranked_results, scores_by_query = cosine_similarity(joined_queries, eligible, work_mat, auth_mat, positive_query_works, query_works, positive_query_authors, query_authors, works, authors)


    final_results_list = []
    for i, result in enumerate(ranked_results[:100]):
        work_data = works[ranked_results[i][0]]
        rankings_data_dict = {
            "title":work_data["title"],
            "author":work_data["author_names"],
            "ranking":i+1,
            "book_url":work_data["url"],
            "image_url":work_data["image"],
            "description":work_data["description"],
            "input_sims": scores_by_query[result[0]]
        }
        final_results_list.append(rankings_data_dict)
    return final_results_list