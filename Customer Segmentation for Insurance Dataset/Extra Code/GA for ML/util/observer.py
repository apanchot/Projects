from util.terminal import Terminal, FontColor
import sys

class LocalSearcObserver():
    def __init__(self, local_search_algorithm):
        self._algorithm = local_search_algorithm
    
    def update( self ):
        state = self._algorithm.get_state()

        # {
        #    "problem"   : self._problem_instance,
        #    "iteration" : self._iteration,
        #    "message"   : message,
        #    "solution"  : self._solution,
        #    "neighbor"  : self._neighbor
        # }

        message = ""
        if "message" in state: message = state["message"]

        # started
        if message == LocalSearchMessage.Started:
            Terminal.clear()
            Terminal.print_box( messages = [ self._algorithm.name, self._algorithm.description ] , font_color = FontColor.Green)
            Terminal.print_box( messages = [ "Problem: " + self._algorithm.problem.name ], font_color = FontColor.Yellow)

        # initialized
        elif message == LocalSearchMessage.Initialized:

            Terminal.print_box( messages = [ "Initialized" ] )
            Terminal.print_line( message= "Initial Solution:") 
            print(f"   Solution: {self._algorithm.solution.representation} - fitness: {self._algorithm.solution.fitness}")
            Terminal.print_box( messages = [ "Iterations"])

        elif message == LocalSearchMessage.ReplacementAccepted:
            iteration = -1
            if "iteration" in state: 
                iteration = state["iteration"]

            Terminal.print_line( message = f"Iteration {iteration:10d} | Solution: {self._algorithm.solution.representation } | Fitness: {self._algorithm.solution.fitness}")

        elif message == LocalSearchMessage.ReplacementRejected:
            iteration = -1
            if "iteration" in state: 
                iteration = state["iteration"]
            
            Terminal.print_line( message = f"Iteration {iteration:10d} | Solution: {self._algorithm.solution.representation } | Fitness: {self._algorithm.solution.fitness} *** (no change)")


        elif message == LocalSearchMessage.StoppedPrematurely:
            Terminal.print_box( messages = ["Stopped Prematurely!"], font_color = FontColor.Yellow )
        
        elif message == LocalSearchMessage.Stopped:
            Terminal.print_box( messages = ["Stopped Max Iterations!"])

        elif message == LocalSearchMessage.StoppedTargetAchieved:
            Terminal.print_box( messages = ["Stopped Target Achieved!"], font_color = FontColor.Green)


class LocalSearchMessage:
    Started                 = "STARTED" 
    Initialized             = "INITIALIZED" 
    ReplacementAccepted     = "REPLACEMENT ACCEPTED"
    ReplacementRejected     = "REPLACEMENT REJECTED"
    StoppedPrematurely      = "STOPPED PREMATURELY"
    Stopped                 = "STOPPED"
    StoppedTargetAchieved   = "STOPPED TARGET ACHIEVED"
