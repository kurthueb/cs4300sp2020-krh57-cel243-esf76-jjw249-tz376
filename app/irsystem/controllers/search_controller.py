from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app import app as data_pool
import json
from app.irsystem.models.search import get_doc_rankings
#import os
#import psutil
project_name = "Book Club"
net_id = "Caroline Lui: cel243, Elisabeth Finkel: esf76, Janie Walter: jjw249, Kurt Huebner: krh57, Taixiang(Max) Zeng: tz376"

# def memory_usage_psutil():
# 	# return the memory usage in MB
# 	process = psutil.Process(os.getpid())
# 	print(process.memory_info().rss/float(2 ** 20))

### helpers ###
def _get_book_from_partial(book_str):
	"""Given a partial string `book_str`, returns a list of elements 
	("book_name by author", work_id, cover) where book_str is a substring
	of book_name. The HTML template can then use the strings, work_ids, 
	and cover image urls to display possible matches for the user to
	select between.
	"""
	AUTOFILL_LIMIT = 20
	works = data_pool.data['works']
	relv_work_ids = []
	relv_books = []
	book_str = book_str.lower()
	for work_id in range(len(works)): # first loop for initial matches
		if len(relv_work_ids) >= AUTOFILL_LIMIT:
			break
		title = works[work_id]['title']
		if book_str == title.lower()[:len(book_str)]:
			relv_work_ids.append(work_id)
	for work_id in range(len(works)): # second loop for substring matches
		if len(relv_work_ids) >= AUTOFILL_LIMIT:
			break
		title = works[work_id]['title']
		if book_str in title.lower() and work_id not in relv_work_ids:
			relv_work_ids.append(work_id)
	for work_id in relv_work_ids:
		authors = works[work_id].get("author_names", ["(unknown)"])
		authors = ", ".join(authors)
		title = works[work_id]['title']
		string = f'"{title}" by {authors}'
		relv_books.append(
			{"string": string, "work_id": work_id, "image": works[work_id].get("image")}
		)
	return relv_books

def _clean(s):
	""" To clean author names """
	return s.lower().replace(".", "").strip()

def _get_author_from_partial(auth_str):
	"""Given a partial string `auth_str`, returns a list of elements 
	(full_name, auth_id) where auth_id has name full_name and where
	auth_str is a substring of full_name.
	
	The HTML template can then use the strings to display possible matches
	for the user to select between.
	"""
	AUTOFILL_LIMIT = 20
	authors = data_pool.data['authors']
	relv_auths = []
	relv_auth_ids = []
	auth_str = _clean(auth_str)
	for i, auth_name in enumerate(authors):
		if len(relv_auth_ids) >= AUTOFILL_LIMIT:
			break
		if auth_str == _clean(auth_name)[:len(auth_str)]:
			relv_auth_ids.append(i)
	for i, auth_name in enumerate(authors):
		if len(relv_auth_ids) >= AUTOFILL_LIMIT:
			break
		if auth_str in _clean(auth_name) and i not in relv_auth_ids:
			relv_auth_ids.append(i)
	for i in relv_auth_ids:
		relv_auths.append({"name": authors[i], "author_id": i})
	return relv_auths

def rescore(inputs, idid):
	"""Given some inputs in form [{work_id: stars}], rescale and
	return rescored inpust in form [{`idid`: work_id, "score": score}]"""
	rescale = {1: -1, 2: -0.5, 3: 0.5, 4: 1, 5: 2}
	rescored = []
	for i in inputs:
		iid, stars = list(i.items())[0]
		if type(iid)==str:
			iid = int(iid)
		if type(stars)==str:
			stars = int(stars)
		rescored.append({idid: iid, "score": rescale.get(stars, 0)})
	return rescored

def _get_reccs(work_ids, auth_ids, desired_genres, excluded_genres):
	works = data_pool.data['works']
	eligible = []
	des, exc = set(desired_genres), set(excluded_genres)
	for i, work in enumerate(works):
		# # Logical AND and logical NOT:
		# if set(work['genres']).issubset(req) and set(work['genres']).isdisjoint(exc):
		# # Logical OR and logical NOT:
		if (set(work['genres'])&des or len(desired_genres)==0) and set(work['genres']).isdisjoint(exc):
			eligible.append(i)
	results = get_doc_rankings(
		work_ids,
		eligible,
		auth_ids,
		data_pool.data['work_mat'],
		data_pool.data['auth_mat'],
		data_pool.data['works'],
		data_pool.data['authors']
	)
	for r in results:
		r['author'] = ", ".join(r['author'])
	return results


### ajax endpoints ###

# GET request to search for matching book names
@irsystem.route('/booknames', methods=['GET'])
def get_book_from_partial():
	partial = request.args.get('partial')
	if not partial:
		return json.dumps([])
	return json.dumps(_get_book_from_partial(partial))


# GET request to search for matching author names
@irsystem.route('/authornames', methods=['GET'])
def get_author_from_partial():
	partial = request.args.get('partial')
	if not partial:
		return json.dumps([])
	return json.dumps(_get_author_from_partial(partial))


### html endpoints ###

# Endpoint that receives preferences
@irsystem.route('/result', methods=['POST'])
def get_reccs():
	req = json.loads(request.data)
	works = req.get('works') # list of dicts: [{work_id: 234, stars: 2}, ...]
	works_rescored = rescore(works, "work_id")
	authors = req.get('authors')
	authors_rescored = rescore(authors, "auth_id")
	req_genres = req.get('required_genres', [])
	ex_genres = req.get('excluded_genres', [])

	results = _get_reccs(works_rescored, authors_rescored, req_genres, ex_genres)
	return json.dumps(results)


# Endpoint to redirect to result page
@irsystem.route('/result', methods=['GET'])
def get_result():
	return render_template('result.html'), 200


# Landing page
@irsystem.route('/', methods=['GET'])
def select():
	return render_template('select.html'), 200
