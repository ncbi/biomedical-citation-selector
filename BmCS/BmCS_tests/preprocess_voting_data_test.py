"""
Module to process testing data for ensemble
"""

def preprocess_data(citations):
    """
    Preprocess data for testing voting model.

    Return dictionary of lists of each feature.
    This testing module is the same as the production module, 
    the only difference being that the variable labels 
    is returned.
    """

    pmids, titles, abstracts, affiliations, journal_nlmid, labels = [], [], [], [], [], []

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
        labels.append(citation['is_indexed'])

    citations = {'abstract': abstracts, 'titles': titles, 'author_list': affiliations}

    return citations, journal_nlmid, labels


