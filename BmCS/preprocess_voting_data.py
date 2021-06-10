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
        
        titles.append(citation['title'])
        abstracts.append(citation['abstract'])
        affiliations.append(citation['affiliations'])

        journal_nlmid.append(citation['journal_nlmid'])

    citations = {'abstract': abstracts, 'titles': titles, 'author_list': affiliations}

    return citations, journal_nlmid, pmids