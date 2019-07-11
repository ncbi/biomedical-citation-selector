"""
Module for ensemble data processing
"""

def preprocess_data(citations):
    """
    Preprocess data for voting model.

    Return dictionary of lists of each feature
    """

    pmids, titles, abstracts, affiliations, journal_nlmid = [], [], [], [], []

    for citation in citations:
        pmids.append(citation['pmid'])
        if citation['title'] == "":
            titles.append("None")
        else:
            titles.append(citation['title'])
        if citation['abstract'] == "":
            abstracts.append("None")
        else:
            abstracts.append(citation['abstract'])
        if citation['author_list'] == "":
            affiliations.append("None")
        else:
            affiliations.append(citation['author_list'])

        journal_nlmid.append(citation['journal_nlmid'])

    citations = {'abstract': abstracts, 'titles': titles, 'author_list': affiliations}

    return citations, journal_nlmid, pmids


