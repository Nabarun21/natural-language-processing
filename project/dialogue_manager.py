import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
import numpy as np
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import utils 
import pickle
import logging
import joblib
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = utils.load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".jlib")
        thread_ids, thread_embeddings = utils.unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.

        question_vec = utils.question_to_vec(question,self.word_embeddings,self.embeddings_dim)
        question_vec=np.float32(question_vec.reshape(1,-1))
        best_thread = pairwise_distances_argmin(question_vec,thread_embeddings)

        del thread_embeddings
        return thread_ids[best_thread][0]


class DialogueManager(object):
    def __init__(self, paths=utils.RESOURCE_PATH):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = utils.unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = utils.unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = utils.unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

        self.chatbot=ChatBot('tiksbot')
        trainer = ChatterBotCorpusTrainer(self.chatbot)
        trainer.train("chatterbot.corpus.english")
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = utils.text_prepare(question)

        # Intent recognition:

        features =self.tfidf_vectorizer.transform([prepared_question])

        intent=self.intent_recognizer.predict(features)

        print(intent)
        # Chit-chat part:   
        if intent == 'dialogue':
    
            return self.chatbot.get_response(question)
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
        
            tag = self.tag_classifier.predict(features)[0]
            print(tag)
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id =self.thread_ranker.get_best_thread(prepared_question,tag)

            return self.ANSWER_TEMPLATE % (tag, thread_id)

