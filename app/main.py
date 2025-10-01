#Steamlit UI
'''
Application entry point (e.g., for a Streamlit, Flask, or simple Tkinter UI). 
Initializes the backend pipeline and launches the visual interface.
'''
import streamlit as st
from core import cube, beginner_solver
from services import pipeline