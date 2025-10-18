#Steamlit UI
'''
Application entry point (e.g., for a Streamlit, Flask, or simple Tkinter UI). 
Initializes the backend pipeline and launches the visual interface.
'''
import streamlit as st
from core import cube, beginner_solver
from services import pipeline

def main():
    st.title("Rubik's Cube Solver")
    st.write("Enter a scramble to solve the cube")
    scramble = st.text_input("Scramble")
    if st.button("Solve"):
        solution = pipeline.run_pipeline(scramble)
        st.write(solution)

if __name__ == "__main__":
    main()