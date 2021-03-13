
import RPi.GPIO as GPIO

#set GPIO numbering mode and define output pins


class Motorize:
    theG = GPIO

    def Start_Engine(self):
        self.theG.setmode(self.theG.BOARD)
        self.theG.setup(7,self.theG.OUT)
        self.theG.setup(11,self.theG.OUT)
        self.theG.setup(13,self.theG.OUT)
        self.theG.setup(15,self.theG.OUT)

    def Key_Up(self):
        self.theG.output(7,False)
        self.theG.output(11,True)
        self.theG.output(13,False)
        self.theG.output(15,True)
    def Key_Down(self):
        self.theG.output(7,True)
        self.theG.output(11,False)
        self.theG.output(13,True)
        self.theG.output(15,False)
    def Key_Right(self):
        self.theG.output(7,True)
        self.theG.output(11,False)
        self.theG.output(13,False)
        self.theG.output(15,True)
    def Key_Left(self):
        self.theG.output(7,False)
        self.theG.output(11,True)
        self.theG.output(13,True)
        self.theG.output(15,False)
    def Key_Enter(self):
        self.theG.output(7,False)
        self.theG.output(11,False)
        self.theG.output(13,False)
        self.theG.output(15,False)

    def Run_Forward(self):
        self.Key_Up()

    def Press_Break(self):
        self.Key_Enter()

    def Stop_Engine(self):
        self.theG.cleanup()
    
