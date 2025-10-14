#beginner_solver
'''
solve(cube_state): Main function that orchestrates the stages. 
Stage Methods: Six separate methods (e.g., stage_white_cross, stage_f2l, etc.) 
implementing the LBL (Layer by Layer) method. Each stage method returns a move sequence and a text explanation.
'''

from core.cube import Cube
from core.notation import Face

class BeginnerSolver:
    def __init__(self, cube: Cube):
        self.cube = cube
        self.solution_moves = []
        self.explanations = []

    # ========== Main Entry Point ==========
    def solve(self):
        """Run all stages sequentially."""
        stages = [
            self.cross,
            self.f2l,
            self.topCross,
            self.getfish,
            self.bOLL,
            self.bPLL
        ]

        for stage_fn in stages:
            if (self.cube.is_solved()):
                break
            stage_fn()
        
        self.solution_moves = self.simplify_moves(self.solution_moves)
        return self.solution_moves

    def simplify_moves(self, moves_list):
        new_list = []
        prev_move = ""
        yCount = 0
        for move in moves_list:
            if move == "yi" or move == "y'":
                yCount += 1
                yCount %= 4
                continue
            if move == "y":
                yCount += 3
                yCount %= 4
                continue
            if move == "Y2":
                yCount += 2
                yCount %= 4
                continue
            if yCount > 0:
                for i in range(yCount):
                    move = self.yTransform(move)
            if prev_move == "" or prev_move == '':
                prev_move = move
                new_list.append(move)
                continue
            if move[0] == prev_move[0]:
                if len(move) == 1:
                    if len(prev_move) <= 1:
                        del new_list[-1]
                        mv = move[0] + "2"
                        new_list.append(mv)
                        prev_move = mv
                        continue
                    if prev_move[1] == "i" or prev_move[1] == "'":
                        del new_list[-1]
                        prev_move = new_list[-1] if len(new_list) > 0 else ""
                        continue
                    if prev_move[1] == "2":
                        del new_list[-1]
                        mv = move[0] + "i"
                        new_list.append(mv)
                        prev_move = mv
                        continue
                if move[1] == "i" or move[1] == "'":
                    if len(prev_move) == 1:
                        del new_list[-1]
                        prev_move = new_list[-1] if len(new_list) > 0 else ""
                        continue
                    if prev_move[1] == "i" or prev_move[1] == "'":
                        del new_list[-1]
                        mv = move[0] + "2"
                        new_list.append(mv)
                        prev_move = mv
                        continue
                    if prev_move[1] == "2":
                        del new_list[-1]
                        mv = move[0]
                        new_list.append(mv)
                        prev_move = mv
                        continue
                if move[1] == "2":
                    if len(prev_move) == 1:
                        del new_list[-1]
                        mv = move[0] + "i"
                        new_list.append(mv)
                        prev_move = mv
                        continue
                    if prev_move[1] == "i" or prev_move[1] == "'":
                        del new_list[-1]
                        mv = move[0]
                        new_list.append(mv)
                        prev_move = mv
                        continue
                    if prev_move[1] == "2":
                        del new_list[-1]
                        prev_move = new_list[-1] if len(new_list) > 0 else ""
                        continue
            new_list.append(move)
            prev_move = move
        solution_length = len(new_list)
        moves_list = new_list
        return moves_list

    def yTransform(self, move):
        if move[0] in ["U", "D"]:
            return move
        if move[0] == "F":
            return "R" + move[1:]
        if move[0] == "R":
            return "B" + move[1:]
        if move[0] == "B":
            return "L" + move[1:]
        if move[0] == "L":
            return "F" + move[1:]
        raise Exception("Invalid move to yTransform: " + move)
    # ========== Individual Stages ==========
    def orient_white_up(self):
        """
        Rotates the cube until the white center is on the U (up) face.
        """
        # Find which face currently has the white center
        for face in Face:
            if self.cube.state[face.value][1][1] == 'W':
                white_face = face
                break

        # Rotate cube so that white face becomes U
        if white_face == Face.D:
            self.cube.rotate_x()
            self.cube.rotate_x()
        elif white_face == Face.F:
            self.cube.rotate_x()
        elif white_face == Face.B:
            self.cube.rotate_x_prime()
        elif white_face == Face.L:
            self.cube.rotate_y()
            self.cube.rotate_x()
        elif white_face == Face.R:
            self.cube.rotate_y_prime()
            self.cube.rotate_x()

        # Now white center should be on U
        assert self.cube.state[Face.U.value][1][1] == 'W'

    def orient_white_down(self):
        """
        Rotates the cube until the white center is on the D (bottom) face.
        """
        # Find which face currently has the white center
        for face in Face:
            if self.cube.state[face.value][1][1] == 'W':
                white_face = face
                break

        # Rotate cube so that white face becomes D
        if white_face == Face.U:
            self.cube.rotate_x()
            self.cube.rotate_x()
        elif white_face == Face.F:
            self.cube.rotate_x_prime()
        elif white_face == Face.B:
            self.cube.rotate_x()
        elif white_face == Face.L:
            self.cube.rotate_y_prime()
            self.cube.rotate_x()
        elif white_face == Face.R:
            self.cube.rotate_y()
            self.cube.rotate_x()

        # Now white center should be on D
        assert self.cube.state[Face.D.value][1][1] == 'W'

    def doMove(self, moves, printCube=False):
        self.cube.apply_moves(moves)  #bring out back-right edge
        self.solution_moves.extend(moves.split())
        # print(moves)
        if printCube:
            self.cube.print_cube()

    def putCrossEdge(self):
        for i in range(3):
            if i == 1:
                self.doMove("Ri U R F F")  #bring out back-right edge
            elif i == 2:
                self.doMove("L Ui Li F F") #bring out back-left edge
            for j in range(4):
                for k in range(4):
                    if "Y" in [self.cube.state[4][0][1], self.cube.state[1][2][1]]:
                        return
                    self.doMove("F", False)
                self.doMove("U", False)

    #Performs the first step of the solution: the cross
    def cross(self):
        self.orient_white_up()
        a = self.cube.state
        for i in range(4):
            self.putCrossEdge()
            assert "Y" in [a[4][0][1], a[1][2][1]]
            if a[1][2][1] == "Y":
                self.doMove("Fi R U Ri F F")   #orient if necessary
            self.doMove("Di", False)

        #permute to correct face: move down face until 2 are lined up,
        #then swap the other 2 if they need to be swapped
        condition = False
        while not condition:
            fSame = a[1][1][1] == a[1][2][1]
            rSame = a[2][1][1] == a[2][1][2]
            bSame = a[5][1][1] == a[5][0][1]
            lSame = a[3][1][1] == a[3][1][0]
            condition = (fSame, rSame, bSame, lSame).count(True) >= 2
            if not condition:
                self.doMove("D")
        if (fSame, rSame, bSame, lSame).count(True) == 4:
            return
        assert (fSame, rSame, bSame, lSame).count(True) == 2
        if not fSame and not bSame:
            self.doMove("F F U U B B U U F F") #swap front-back
        elif not rSame and not lSame:
            self.doMove("R R U U L L U U R R") #swap right-left
        elif not fSame and not rSame:
            self.doMove("F F Ui R R U F F") #swap front-right
        elif not rSame and not bSame:
            self.doMove("R R Ui B B U R R") #swap right-back
        elif not bSame and not lSame:
            self.doMove("B B Ui L L U B B") #swap back-left
        elif not lSame and not fSame:
            self.doMove("L L Ui F F U L L") #swap left-front
        fSame = a[1][1][1] == a[1][2][1]
        rSame = a[2][1][1] == a[2][1][2]
        bSame = a[5][1][1] == a[5][0][1]
        lSame = a[3][1][1] == a[3][1][0]
        assert all([fSame, rSame, bSame, lSame])

    #This is uses all the f2l algs to solve all the cases possible
    def solveFrontSlot(self):
        a = self.cube.state
        #This will be F2L, with all 42 cases
        rmid = a[2][1][1]
        fmid = a[1][1][1]
        dmid = a[4][1][1]
        #corner orientations if in U layer, first letter means the direction that the color is facing
        fCorU = a[1][0][2] == dmid and a[0][2][2] == fmid and a[2][2][0] == rmid
        rCorU = a[2][2][0] == dmid and a[1][0][2] == fmid and a[0][2][2] == rmid
        uCorU = a[0][2][2] == dmid and a[2][2][0] == fmid and a[1][0][2] == rmid
        #Corner orientations for correct location in D layer
        fCorD = a[1][2][2] == dmid and a[2][2][2] == fmid and a[4][0][2] == rmid
        rCorD = a[2][2][2] == dmid and a[4][0][2] == fmid and a[1][2][2] == rmid
        dCorD = a[4][0][2] == dmid and a[1][2][2] == fmid and a[2][2][2] == rmid #This is solved spot
        #edge orientations on U layer, normal or flipped version based on F face
        norEdgeFU = a[1][0][1] == fmid and a[0][2][1] == rmid
        norEdgeLU = a[3][1][2] == fmid and a[0][1][0] == rmid
        norEdgeBU = a[5][2][1] == fmid and a[0][0][1] == rmid
        norEdgeRU = a[2][1][0] == fmid and a[0][1][2] == rmid
        norEdgeAny = norEdgeFU or norEdgeLU or norEdgeBU or norEdgeRU
        flipEdgeFU = a[0][2][1] == fmid and a[1][0][1] == rmid
        flipEdgeLU = a[0][1][0] == fmid and a[3][1][2] == rmid
        flipEdgeBU = a[0][0][1] == fmid and a[5][2][1] == rmid
        flipEdgeRU = a[0][1][2] == fmid and a[2][1][0] == rmid
        flipEdgeAny = flipEdgeFU or flipEdgeLU or flipEdgeBU or flipEdgeRU
        #edge orientations for normal or flipped insertion into slot
        norEdgeInsert = a[1][1][2] == fmid and a[2][2][1] == rmid #This is solved spot
        flipEdgeInsert = a[2][2][1] == fmid and a[1][1][2] == rmid
        #these are for if the back right or front left slots are open or not
        backRight = a[4][2][2] == dmid and a[5][1][2] == a[5][0][2] == a[5][1][1] and a[2][0][1] == a[2][0][2] == rmid
        frontLeft = a[4][0][0] == dmid and a[1][1][0] == a[1][2][0] == fmid and a[3][2][0] == a[3][2][1] == a[3][1][1]
        
        if dCorD and norEdgeInsert: 
            return
        #Easy Cases
        elif fCorU and flipEdgeRU: #Case 1
            self.doMove("U R Ui Ri")
        elif rCorU and norEdgeFU: #Case 2
            self.doMove("F Ri Fi R")
        elif fCorU and norEdgeLU: #Case 3
            self.doMove("Fi Ui F")
        elif rCorU and flipEdgeBU: #Case 4
            self.doMove("R U Ri")
        #Reposition Edge
        elif fCorU and flipEdgeBU: #Case 5
            self.doMove("F2 Li Ui L U F2")
        elif rCorU and norEdgeLU: #Case 6
            self.doMove("R2 B U Bi Ui R2")
        elif fCorU and flipEdgeLU: #Case 7
            self.doMove("Ui R U2 Ri U2 R Ui Ri")
        elif rCorU and norEdgeBU: #Case 8
            self.doMove("U Fi U2 F Ui F Ri Fi R")
        # Reposition edge and Corner Flip
        elif fCorU and norEdgeBU: #Case 9
            self.doMove("Ui R Ui Ri U Fi Ui F")
        elif rCorU and flipEdgeLU: #Case 10
            if not backRight:
                self.doMove("Ri U R2 U Ri")
            else:
                self.doMove("Ui R U Ri U R U Ri")
        elif fCorU and norEdgeRU: #Case 11
            self.doMove("Ui R U2 Ri U Fi Ui F")
        elif rCorU and flipEdgeFU: # Case 12
            if not backRight:
                self.doMove("Ri U2 R2 U Ri")
            else:
                self.doMove("Ri U2 R2 U R2 U R")
        elif fCorU and norEdgeFU: #Case 13
            if not backRight:
                self.doMove("Ri U R Fi Ui F")
            else:
                self.doMove("U Fi U F Ui Fi Ui F")
        elif rCorU and flipEdgeRU: #Case 14
            self.doMove("Ui R Ui Ri U R U Ri")
        # Split Pair by Going Over
        elif fCorU and flipEdgeFU: #Case 15
            if not backRight:
                self.doMove("Ui Ri U R Ui R U Ri")
            elif not frontLeft:
                self.doMove("U R Ui Ri D R Ui Ri Di")
            else:
                self.doMove("U Ri F R Fi U R U Ri")
        elif rCorU and norEdgeRU: # Case 16
            self.doMove("R Ui Ri U2 Fi Ui F")
        elif uCorU and flipEdgeRU: #Case 17
            self.doMove("R U2 Ri Ui R U Ri")
        elif uCorU and norEdgeFU: # Case 18
            self.doMove("Fi U2 F U Fi Ui F")
        # Pair made on side
        elif uCorU and flipEdgeBU: #Case 19
            self.doMove("U R U2 R2 F R Fi")
        elif uCorU and norEdgeLU: #Case 20
            self.doMove("Ui Fi U2 F2 Ri Fi R")
        elif uCorU and flipEdgeLU: #Case 21
            self.doMove("R B U2 Bi Ri")
        elif uCorU and norEdgeBU: #Case 22
            self.doMove("Fi Li U2 L F")
        #Weird Cases
        elif uCorU and flipEdgeFU: #Case 23
            self.doMove("U2 R2 U2 Ri Ui R Ui R2")
        elif uCorU and norEdgeRU: #Case 24
            self.doMove("U Fi Li U L F R U Ri")
        #Corner in Place, edge in the U face (All these cases also have set-up moves in case the edge is in the wrong orientation
        elif dCorD and flipEdgeAny: #Case 25
            if flipEdgeBU:
                self.doMove("U") #set-up move
            elif flipEdgeLU:
                self.doMove("U2") #set-up move
            elif flipEdgeFU:
                self.doMove("Ui") #set-up move
            if not backRight:
                self.doMove("R2 Ui Ri U R2")
            else:
                self.doMove("Ri Fi R U R Ui Ri F")
        elif dCorD and norEdgeAny: #Case 26
            if norEdgeRU:
                self.doMove("U") #set-up move
            elif norEdgeBU:
                self.doMove("U2") #set-up move
            elif norEdgeLU:
                self.doMove("Ui") #set-up move
            self.doMove("U R Ui Ri F Ri Fi R")
        elif fCorD and flipEdgeAny: #Case 27
            if flipEdgeBU:
                self.doMove("U") #set-up move
            elif flipEdgeLU:
                self.doMove("U2") #set-up move
            elif flipEdgeFU:
                self.doMove("Ui") #set-up move
            self.doMove("R Ui Ri U R Ui Ri")
        elif rCorD and norEdgeAny: #Case 28
            if norEdgeRU:
                self.doMove("U") #set-up move
            elif norEdgeBU:
                self.doMove("U2") #set-up move
            elif norEdgeLU:
                self.doMove("Ui") #set-up move
            self.doMove("R U Ri Ui F Ri Fi R")
        elif fCorD and norEdgeAny: #Case 29
            if norEdgeRU:
                self.doMove("U") #set-up move
            elif norEdgeBU:
                self.doMove("U2") #set-up move
            elif norEdgeLU:
                self.doMove("Ui") #set-up move
            self.doMove("U2 R Ui Ri Fi Ui F")
        elif rCorD and flipEdgeAny: #Case 30
            if flipEdgeBU:
                self.doMove("U") #set-up move
            elif flipEdgeLU:
                self.doMove("U2") #set-up move
            elif flipEdgeFU:
                self.doMove("Ui") #set-up move
            self.doMove("R U Ri Ui R U Ri")
        #Edge in place, corner in U Face
        elif uCorU and flipEdgeInsert: # Case 31
            self.doMove("R U2 Ri Ui F Ri Fi R")
        elif uCorU and norEdgeInsert: # Case 32
            self.doMove("R2 U R2 U R2 U2 R2")
        elif fCorU and norEdgeInsert: # Case 33
            self.doMove("Ui R Ui Ri U2 R Ui Ri")
        elif rCorU and norEdgeInsert: # Case 34
            self.doMove("Ui R U2 Ri U R U Ri")
        elif fCorU and flipEdgeInsert: # Case 35
            self.doMove("U2 R Ui Ri Ui Fi Ui F")
        elif rCorU and flipEdgeInsert: # Case 36
            self.doMove("U Fi Ui F Ui R U Ri")
        #Edge and Corner in place
        #Case 37 is Lol case, already completed
        elif dCorD and flipEdgeInsert: #Case 38 (Typical flipped f2l pair case
            self.doMove("R2 U2 F R2 Fi U2 Ri U Ri")
        elif fCorD and norEdgeInsert: # Case 39
            self.doMove("R2 U2 Ri Ui R Ui Ri U2 Ri")
        elif rCorD and norEdgeInsert: # Case 40
            self.doMove("R U2 R U Ri U R U2 R2")
        elif fCorD and flipEdgeInsert: #Case 41
            self.doMove("F2 Li Ui L U F Ui F")
        elif rCorD and flipEdgeInsert: # Case 42
            self.doMove("R Ui Ri Fi Li U2 L F")

    #Returns true if the f2l Corner in FR spot is inserted and oriented correctly
    def f2lCorner(self):
        a = self.cube.state
        return a[4][0][2] == a[4][1][1] and a[1][2][2] == a[1][1][1] and a[2][2][2] == a[2][1][1] #This is solved spot

    #Returns true if the f2l edge in FR spot is inserted and oriented correctly    
    def f2lEdge(self):
        a = self.cube.state
        return a[1][1][2] == a[1][1][1] and a[2][2][1] == a[2][1][1] #This is solved spot

    #Returns true if the f2l edge and corner are properly inserted and orientated in the FR position
    def f2lCorrect(self):
        a = self.cube.state
        return self.f2lCorner() and self.f2lEdge()

    # returns if the f2l edge is on the top layer at all
    def f2lEdgeOnTop(self):
        a = self.cube.state
        rmid = a[2][1][1]
        fmid = a[1][1][1]
        dmid = a[4][1][1]
        #edge orientations on U layer, normal or flipped version based on F face
        norEdgeFU = a[1][0][1] == fmid and a[0][2][1] == rmid
        norEdgeLU = a[3][1][2] == fmid and a[0][1][0] == rmid
        norEdgeBU = a[5][2][1] == fmid and a[0][0][1] == rmid
        norEdgeRU = a[2][1][0] == fmid and a[0][1][2] == rmid
        norEdgeAny = norEdgeFU or norEdgeLU or norEdgeBU or norEdgeRU
        flipEdgeFU = a[0][2][1] == fmid and a[1][0][1] == rmid
        flipEdgeLU = a[0][1][0] == fmid and a[3][1][2] == rmid
        flipEdgeBU = a[0][0][1] == fmid and a[5][2][1] == rmid
        flipEdgeRU = a[0][1][2] == fmid and a[2][1][0] == rmid
        flipEdgeAny = flipEdgeFU or flipEdgeLU or flipEdgeBU or flipEdgeRU
        return norEdgeAny or flipEdgeAny

    #returns true if the f2l edge is inserted. Can be properly orientated, or flipped.
    def f2lEdgeInserted(self):
        a = self.cube.state
        rmid = a[2][1][1]
        fmid = a[1][1][1]
        #edge orientations for normal or flipped insertion into slot
        norEdgeInsert = a[1][1][2] == fmid and a[2][2][1] == rmid #This is solved spot
        flipEdgeInsert = a[2][2][1] == fmid and a[1][1][2] == rmid
        return norEdgeInsert or flipEdgeInsert

    #This is used to determine if the front f2l edge is inserted or not, the parameter is for the requested edge. takes BR, BL, and FL as valid
    def f2lEdgeInserted2(self, p):
        a = self.cube.state
        rmid = a[2][1][1]
        fmid = a[1][1][1]
        #edge orientations for normal or flipped insertion into slot
        norEdgeInsert = a[1][1][2] == fmid and a[2][2][1] == rmid #This is solved spot
        flipEdgeInsert = a[2][2][1] == fmid and a[1][1][2] == rmid
        #Edge orientations in comparison to Front and Right colors
        BR = (a[5][1][2] == fmid and a[2][0][1] == rmid) or (a[5][1][2] == rmid and a[2][0][1] == fmid)
        BL = (a[3][0][1] == fmid and a[5][1][0] == rmid) or (a[3][0][1] == rmid and a[5][1][0] == fmid)
        FL = (a[3][2][1] == fmid and a[1][1][0] == rmid) or (a[3][2][1] == rmid and a[1][1][0] == fmid)
        
        if p == "BR":
            if BR:
                return True
            else:
                return False
        elif p == "BL":
            if BL:
                return True
            return False
        elif p == "FL":
            if FL:
                return True
            return False
        elif p == "FR":
            if norEdgeInsert or flipEdgeInsert:
                return True
        return False

    #returns true if f2l corner is inserted, doesn't have to be orientated correctly
    def f2lCornerInserted(self):
        a = self.cube.state
        rmid = a[2][1][1]
        fmid = a[1][1][1]
        dmid = a[4][1][1]
        #Corner orientations for correct location in D layer
        fCorD = a[1][2][2] == dmid and a[2][2][2] == fmid and a[4][0][2] == rmid
        rCorD = a[2][2][2] == dmid and a[4][0][2] == fmid and a[1][2][2] == rmid
        dCorD = a[4][0][2] == dmid and a[1][2][2] == fmid and a[2][2][2] == rmid #This is solved spot
        return fCorD or rCorD or dCorD

    #Returns true if there is an f2l corner located in the FR orientation
    def f2lFRCor(self):
        a = self.cube.state
        rmid = a[2][1][1]
        fmid = a[1][1][1]
        dmid = a[4][1][1]
        #corner orientations if in U layer, first letter means the direction that the color is facing
        fCorU = a[1][0][2] == dmid and a[0][2][2] == fmid and a[2][2][0] == rmid
        rCorU = a[2][2][0] == dmid and a[1][0][2] == fmid and a[0][2][2] == rmid
        uCorU = a[0][2][2] == dmid and a[2][2][0] == fmid and a[1][0][2] == rmid
        return fCorU or rCorU or uCorU

    #Returns true if there is an f2l Edge located in the FU position
    def f2lFUEdge(self):
        a = self.cube.state
        rmid = a[2][1][1]
        fmid = a[1][1][1]
        norEdgeFU = a[1][0][1] == fmid and a[0][2][1] == rmid
        flipEdgeFU = a[0][2][1] == fmid and a[1][0][1] == rmid
        return norEdgeFU or flipEdgeFU

    #returns true if f2l corner is located on the U layer
    def f2lCornerOnTop(self):
        wasFound = False
        for i in range(4): #Does 4 U moves to find the corner
            if self.f2lFRCor():
                wasFound = True
            self.doMove("U")
        return wasFound

    #Will return the loction of the corner that belongs in the FR spot. Either returns BR, BL, FL, or FR.
    def f2lCornerCheck(self):
        r = "FR"
        count = 0
        while count < 4:
            if count == 0:
                if self.f2lCornerInserted():
                    r = "FR"
            elif count == 1:
                if self.f2lCornerInserted():
                    r = "FL"
            elif count == 2:
                if self.f2lCornerInserted():
                    r = "BL"
            elif count == 3:
                if self.f2lCornerInserted():
                    r = "BR"
            self.doMove("D")
            count += 1
        return r

    #Will return the loction of the edge that belongs in the FR spot.
    #Either returns BR, BL, FL, or FR.
    def f2lEdgeCheck(self):
        if self.f2lEdgeInserted2("FL"):
            return "FL"
        elif self.f2lEdgeInserted2("BL"):
            return "BL"
        elif self.f2lEdgeInserted2("BR"):
            return "BR"
        elif self.f2lEdgeInserted2("FR"):
            return "FR"
        else:
            raise Exception("f2lEdgeCheck() Exception")
        
    #This is for the case where the Edge is inserted, but the corner is not
    def f2lEdgeNoCorner(self):
        a = self.cube.state
        topEdgeTop = a[0][2][1]
        topEdgeFront = a[1][0][1]
        rmid = a[2][1][1]
        bmid = a[5][1][1]
        lmid = a[3][1][1]
        fmid = a[1][1][1]
        #This is for comparing the front edge to other various edges for advanced algs/lookahead
        BREdge = (topEdgeTop == rmid or topEdgeTop == bmid) and (topEdgeFront == rmid or topEdgeFront == bmid)
        BLEdge = (topEdgeTop == lmid or topEdgeTop == bmid) and (topEdgeFront == lmid or topEdgeFront == bmid)
        FLEdge = (topEdgeTop == fmid or topEdgeTop == lmid) and (topEdgeFront == fmid or topEdgeFront == lmid)
        if f2lCornerOnTop():
            while True:
                self.solveFrontSlot()
                if f2lCorrect():
                    break
                self.doMove("U")
        else:
            if self.f2lCornerCheck() == "BR":
                if BREdge:
                    self.doMove("Ri Ui R U2")
                else:
                    self.doMove("Ri U R U")
            elif self.f2lCornerCheck() == "BL":
                if BLEdge:
                    self.doMove("L U Li U")
                else:
                    self.doMove("L Ui Li U2")
            elif self.f2lCornerCheck() == "FL":
                if FLEdge:
                    self.doMove("Li U L Ui")
                else:
                    self.doMove("Li Ui L")
        self.solveFrontSlot()

        if not self.f2lCorrect():
            raise Exception("Exception found in f2lEdgeNoCorner()")
   
    #This is the case for if the corner is inserted, but the edge is not
    def f2lCornerNoEdge(self):
        a = self.cube.state
        topEdgeTop = a[0][2][1]
        topEdgeFront = a[1][0][1]
        rmid = a[2][1][1]
        bmid = a[5][1][1]
        lmid = a[3][1][1]
        fmid = a[1][1][1]
        #This is for comparing the front edge to other various edges for advanced algs/lookahead
        BREdge = (topEdgeTop == rmid or topEdgeTop == bmid) and (topEdgeFront == rmid or topEdgeFront == bmid)
        BLEdge = (topEdgeTop == lmid or topEdgeTop == bmid) and (topEdgeFront == lmid or topEdgeFront == bmid)
        FLEdge = (topEdgeTop == fmid or topEdgeTop == lmid) and (topEdgeFront == fmid or topEdgeFront == lmid)
        if self.f2lEdgeOnTop():
            while True:
                self.solveFrontSlot()
                if self.f2lCorrect():
                    break
                self.doMove("U")
        else:
            if self.f2lEdgeCheck() == "BR":
                if BREdge:
                    self.doMove("Ri Ui R U2")
                else:
                    self.doMove("Ri U R U")
            elif self.f2lEdgeCheck() == "BL":
                if BLEdge:
                    self.doMove("L U Li U")
                else:
                    self.doMove("L Ui Li U2")
            elif self.f2lEdgeCheck() == "FL":
                if FLEdge:
                    self.doMove("Li U L Ui")
                else:
                    self.doMove("Li Ui L")
        self.solveFrontSlot()

        if not self.f2lCorrect():
            raise Exception("Exception found in f2lCornerNoEdge()")

    #this is the case for if the corner is on top, and the edge is not. Neither are inserted properly. Edge must be in another slot.
    def f2lCornerTopNoEdge(self):
        a = self.cube.state
        topEdgeTop = a[0][2][1]
        topEdgeFront = a[1][0][1]
        rmid = a[2][1][1]
        bmid = a[5][1][1]
        lmid = a[3][1][1]
        fmid = a[1][1][1]
        #This is for comparing the front edge to other various edges for advanced algs/lookahead
        BREdge = (topEdgeTop == rmid or topEdgeTop == bmid) and (topEdgeFront == rmid or topEdgeFront == bmid)
        BLEdge = (topEdgeTop == lmid or topEdgeTop == bmid) and (topEdgeFront == lmid or topEdgeFront == bmid)
        FLEdge = (topEdgeTop == fmid or topEdgeTop == lmid) and (topEdgeFront == fmid or topEdgeFront == lmid)

        #Turn the top until the corner on the U face is in the proper position
        while True:
            if self.f2lFRCor():
                break
            self.doMove("U")
        #We will be checking additional edges to choose a more fitting alg for the sake of looking ahead
        if self.f2lEdgeCheck() == "BR":
            if BREdge:
                self.doMove("Ri Ui R")
            else:
                self.doMove("Ri U R")
        elif self.f2lEdgeCheck() == "BL":
            if BLEdge:
                self.doMove("U2 L Ui Li")
            else:
                self.doMove("L Ui Li U")
        elif self.f2lEdgeCheck() == "FL":
            if FLEdge:
                self.doMove("U2 Li Ui L U2")
            else:
                self.doMove("Li Ui L U")
        self.solveFrontSlot()

        if not self.f2lCorrect():
            raise Exception("Exception found in f2lCornerTopNoEdge()")

    #This is the case for if the edge is on top, and the corner is not. Neither are inserted properly. Corner must be in another slot.
    #The lookahead for this step is comparing the back edge to the slots, rather than the front one like other cases have
    def f2lEdgeTopNoCorner(self):
        a = self.cube.state
        BackEdgeTop = a[0][0][1]
        BackEdgeBack = a[5][2][1]
        rmid = a[2][1][1]
        bmid = a[5][1][1]
        lmid = a[3][1][1]
        fmid = a[1][1][1]
        rs1 = BackEdgeTop == rmid or BackEdgeTop == bmid
        rs2 = BackEdgeBack == rmid or BackEdgeBack == bmid
        #This is for comparing the back edge to other various edges for advanced algs/lookahead
        BREdge = rs1 and rs2
        BLEdge = (BackEdgeTop == lmid or BackEdgeTop == bmid) and (BackEdgeBack == lmid or BackEdgeBack == bmid)
        FLEdge = (BackEdgeTop == fmid or BackEdgeTop == lmid) and (BackEdgeBack == fmid or BackEdgeBack == lmid)

        #Turn the top until the corner on the U face is in the proper position
        while True:
            if self.f2lFUEdge():
                break
            self.doMove("U")
        #We will be checking additional edges to choose a more fitting alg for the sake of looking ahead
        if self.f2lCornerCheck() == "BR":
            if BREdge:
                self.doMove("Ri U R U")
            else:
                self.doMove("Ui Ri U R U")
        elif self.f2lCornerCheck() == "BL":
            if BLEdge:
                self.doMove("L Ui Li U2")
            else:
                self.doMove("U2 L U2 Li")
        elif self.f2lCornerCheck() == "FL":
            if FLEdge:
                self.doMove("Li Ui L")
            else:
                self.doMove("U Li Ui L")
        self.solveFrontSlot()

        if not self.f2lCorrect():
            raise Exception("Exception found in f2lEdgeTopNoCorner()")

    #This is the case for if the edge or corner are not on top, and not inserted properly. They must both be in other slots.
    def f2lNoEdgeOrCorner(self):
        a = self.cube.state
        #The strategy here is to first find the corner and get it out. I will place it in the FR position where it belongs
        #I will then check if I have a case, and if we are all solved.
        #If I don't have it solved at this point, I will have to follow what happens in f2lCornerTopNoEdge()

        BackEdgeTop = a[0][0][1]
        BackEdgeBack = a[5][2][1]
        rmid = a[2][1][1]
        bmid = a[5][1][1]
        lmid = a[3][1][1]
        fmid = a[1][1][1]
        #This is for comparing the back edge to other various edges for advanced algs/lookahead
        BREdge = (BackEdgeTop == rmid or BackEdgeTop == bmid) and (BackEdgeBack == rmid or BackEdgeBack == bmid)
        BLEdge = (BackEdgeTop == lmid or BackEdgeTop == bmid) and (BackEdgeBack == lmid or BackEdgeBack == bmid)
        FLEdge = (BackEdgeTop == fmid or BackEdgeTop == lmid) and (BackEdgeBack == fmid or BackEdgeBack == lmid)
        
        #We will be checking additional edges to choose a more fitting alg for the sake of looking ahead
        if self.f2lCornerCheck() == "BR":
            if BREdge:
                self.doMove("Ri U R U")
            else:
                self.doMove("Ui Ri U R U")
        elif self.f2lCornerCheck() == "BL":
            if BLEdge:
                self.doMove("L Ui Li U2")
            else:
                self.doMove("U2 L U2 Li")
        elif self.f2lCornerCheck() == "FL":
            if FLEdge:
                self.doMove("Li Ui L")
            else:
                self.doMove("U Li Ui L")
        self.solveFrontSlot()

        if self.f2lCorrect():
            return
        else:
            self.f2lCornerTopNoEdge()

        if not self.f2lCorrect():
            raise Exception("Exception found in f2lNoEdgeOrCorner()")

    #Will return true if the f2l is completed
    def isf2lDone(self):
        a = self.cube.state
        rside = a[2][0][1] == a[2][0][2] == a[2][1][1] == a[2][1][2] == a[2][2][1] == a[2][2][2]
        bside = a[5][0][0] == a[5][0][1] == a[5][0][2] == a[5][1][0] == a[5][1][1] == a[5][1][2]
        lside = a[3][0][0] == a[3][0][1] == a[3][1][0] == a[3][1][1] == a[3][2][0] == a[3][2][1]
        fside = a[1][1][0] == a[1][1][1] == a[1][1][2] == a[1][2][0] == a[1][2][1] == a[1][2][2]
        return rside and bside and lside and fside

    #f2l will solve the first 2 layers, checks for each case, then does a Y move to check the next
    def f2l(self):
        self.orient_white_up()
        f2l_list = []
        pairsSolved = 0
        uMoves = 0
        while pairsSolved < 4:
            if not self.f2lCorrect():
                #while not f2lCorrect():
                while uMoves < 4: #4 moves before checking rare cases
                    self.solveFrontSlot()
                    if self.f2lCorrect():
                        pairsSolved += 1
                        f2l_list.append("Normal Case")
                        break
                    else:
                        f2l_list.append("Scanning")
                        uMoves += 1
                        self.doMove("U")
                if not self.f2lCorrect():
                    if not self.f2lCornerInserted() and self.f2lEdgeInserted():
                        f2l_list.append("Rare case 1")
                        self.f2lEdgeNoCorner()
                        pairsSolved += 1
                    elif not self.f2lEdgeInserted() and self.f2lCornerInserted():
                        f2l_list.append("Rare case 2")
                        self.f2lCornerNoEdge()
                        pairsSolved += 1
                    #At this point, they can't be inserted, must be in U or other spot
                    elif not self.f2lEdgeOnTop() and self.f2lCornerOnTop():
                        f2l_list.append("Rare Case 3")
                        self.f2lCornerTopNoEdge()
                        pairsSolved += 1
                    elif self.f2lEdgeOnTop() and not self.f2lCornerOnTop():
                        f2l_list.append("Rare Case 4")
                        self.f2lEdgeTopNoCorner()
                        self.solveFrontSlot()
                        pairsSolved += 1
                    elif not self.f2lEdgeOnTop() and not self.f2lCornerOnTop():
                        f2l_list.append("Rare Case 5")
                        self.f2lNoEdgeOrCorner()
                        pairsSolved += 1
                    else:
                        raise Exception("f2l Impossible Case Exception")
            else:
                pairsSolved += 1
            f2l_list.append("We have ")
            f2l_list.append(str(pairsSolved))
            uMoves = 0
            self.doMove("y'")
        assert(self.isf2lDone())

    def topCross(self):
        a = self.cube.state
        # if all the edges are all equal to eachother (all being white)
        if a[0][0][1] == a[0][1][0] == a[0][1][2] == a[0][2][1]:
            #print("Cross already done, step skipped")
            return
        #If this is true, we have our cross and we can go onto the next step
        else:
            while a[0][0][1] != "W" or a[0][1][0] != "W" or a[0][1][2] != "W" or a[0][2][1] != "W":
                if a[0][1][0] == a[0][1][2]:
                        #if we have a horizontal line Just do alg
                        self.doMove("F R U Ri Ui Fi")
                        break #breaking w/o having to recheck while conditions again, this will give us a cross
                elif a[0][0][1] == a[0][2][1]:
                        # if we have a vertical line, do a U then alg
                        self.doMove("U F R U Ri Ui Fi")
                        break
                elif a[0][0][1] != "W" and a[0][1][0] != "W" and a[0][1][2] != "W" and a[0][2][1] != "W":
                        #This would mean we have a dot case, so perform
                        self.doMove("F U R Ui Ri Fi U F R U Ri Ui Fi")
                        break
                elif a[0][1][2] == a[0][2][1] or a[0][0][1] == a[0][1][0]:
                        # If we have an L case in the top left or the bottom right, will give us a line
                        self.doMove("F R U Ri Ui Fi")
                else:
                        #This is we dont have a line, dot, cross, or L in top left or bottom right
                        self.doMove("U")

    def isTopSolved(self):
        a = self.cube.state
        #determines if the top of the cube is solved.
        if a[0][0][0] == a[0][0][1] == a[0][0][2] == a[0][1][0] == a[0][1][1] == a[0][1][2] == a[0][2][0] == a[0][2][1] == a[0][2][2]:
            return True
        else:
            return False	

    def fish(self):
        a = self.cube.state
        return [a[0][0][0], a[0][0][2], a[0][2][0], a[0][2][2]].count(a[0][1][1]) == 1

    def sune(self):
        self.doMove("R U Ri U R U2 Ri")

    def antisune(self):
        self.doMove("R U2 Ri Ui R Ui Ri")

    def getfish(self):
        for i in range(4):
            if self.fish():
                return
            self.sune()
            if self.fish():
                return
            self.antisune()
            self.doMove("U")
        assert self.fish()

    def bOLL(self):
        #self.orient_white_up()
        a = self.cube.state
        self.getfish()
        if self.fish():
            while a[0][0][2] != a[0][1][1]:
                self.doMove("U")
            if a[1][0][0] == a[0][1][1]:
                self.antisune()
            elif a[5][2][0] == a[0][1][1]:
                self.doMove("U2")
                self.sune()
            else:
                raise Exception("Something went wrong")
        else:
            raise Exception("Fish not set up")
        assert self.isTopSolved()

    def getCornerState(self):
        a = self.cube.state
        corner0 = a[1][0][0] == a[1][1][1] and a[3][2][2] == a[3][1][1]
        corner1 = a[1][0][2] == a[1][1][1] and a[2][2][0] == a[2][1][1]
        corner2 = a[5][2][2] == a[5][1][1] and a[2][0][0] == a[2][1][1]
        corner3 = a[5][2][0] == a[5][1][1] and a[3][0][2] == a[3][1][1]
        return [corner0, corner1, corner2, corner3]

    #Does permutation of the top layer corners, orients them properly
    def permuteCorners(self):
        a = self.cube.state
        for i in range(2):
            for j in range(4):
                num = self.getCornerState().count(True)
                if num == 4:
                    return
                if num == 1:
                    index = self.getCornerState().index(True)
                    for k in range(index):
                        self.doMove("yi")
                    if a[1][0][2] == a[2][1][1]:
                        self.doMove("R2 B2 R F Ri B2 R Fi R")
                    else:
                        self.doMove("Ri F Ri B2 R Fi Ri B2 R2")
                    for f in range(index):
                        self.doMove("y")
                    return
                self.doMove("U")
            self.doMove("R2 B2 R F Ri B2 R Fi R")

    #Does permutation of the top layer edges, must be H, Z or U perms after orientation
    def permuteEdges(self):
        a = self.cube.state
        if all(self.getEdgeState()):
            return
        if a[1][0][1] == a[5][1][1] and a[5][2][1] == a[1][1][1]: #H perm
            self.doMove("R2 U2 R U2 R2 U2 R2 U2 R U2 R2")
        elif a[1][0][1] == a[2][1][1] and a[2][1][0] == a[1][1][1]: #Normal Z perm
            self.doMove("U Ri Ui R Ui R U R Ui Ri U R U R2 Ui Ri U")
        elif a[1][0][1] == a[3][1][1] and a[3][1][2] == a[1][1][1]: # Not oriented Z perm
            self.doMove("Ri Ui R Ui R U R Ui Ri U R U R2 Ui Ri U2")
        else:
            uNum = 0
            while True:
                if a[5][2][0] == a[5][2][1] == a[5][2][2]: #solid bar is on back then
                    if a[3][1][2] == a[1][0][0]: #means we have to do counterclockwise cycle
                        self.doMove("R Ui R U R U R Ui Ri Ui R2")
                        break
                    else:
                        self.doMove("R2 U R U Ri Ui Ri Ui Ri U Ri")
                        break
                else:
                    self.doMove("U")
                    uNum += 1
            for x in range(uNum):
                self.doMove("Ui")

    def getEdgeState(self):
        a = self.cube.state
        fEdge = a[1][0][1] == a[1][1][1]
        rEdge = a[2][1][0] == a[2][1][1]
        bEdge = a[5][2][1] == a[5][1][1]
        lEdge = a[3][1][2] == a[3][1][1]
        return [fEdge, rEdge, bEdge, lEdge]
            
    def topCorners(self):
        self.permuteCorners()
        assert all(self.getCornerState())

    def topEdges(self):
        self.permuteEdges()
        assert all(self.getEdgeState())

    def bPLL(self):
        self.topCorners()
        self.topEdges()

    def isSolved(self):
        a = self.cube.state
        uside = a[0][0][0] == a[0][0][1] == a[0][0][2] == a[0][1][0] == a[0][1][1] == a[0][1][2] == a[0][2][0] == a[0][2][1] == a[0][2][2]
        fside = a[1][0][0] == a[1][0][1] == a[1][0][2] == a[1][1][0] == a[1][1][1] == a[1][1][2] == a[1][2][0] == a[1][2][1] == a[1][2][2]
        rside = a[2][0][0] == a[2][0][1] == a[2][0][2] == a[2][1][0] == a[2][1][1] == a[2][1][2] == a[2][2][0] == a[2][2][1] == a[2][2][2]
        lside = a[3][0][0] == a[3][0][1] == a[3][0][2] == a[3][1][0] == a[3][1][1] == a[3][1][2] == a[3][2][0] == a[3][2][1] == a[3][2][2]
        dside = a[4][0][0] == a[4][0][1] == a[4][0][2] == a[4][1][0] == a[4][1][1] == a[4][1][2] == a[4][2][0] == a[4][2][1] == a[4][2][2]
        bside = a[5][0][0] == a[5][0][1] == a[5][0][2] == a[5][1][0] == a[5][1][1] == a[5][1][2] == a[5][2][0] == a[5][2][1] == a[5][2][2]
        return uside and fside and rside and lside and dside and bside
        
