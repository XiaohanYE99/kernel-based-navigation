import numpy as np
import math,pyglet,cv2,os

class Viewer(pyglet.window.Window):
    def __init__(self, wind_size=(800,800), checker=(50,50,[230,230,230])):
        config = pyglet.gl.Config(sample_buffers=1, samples=4)#16
        super(Viewer, self).__init__( resizable=False,
                                     width=wind_size[0], height=wind_size[1], 
                                     caption='Unlabled Multi-Agent Navigation', 
                                     vsync=True)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        # display whole batch at once
        self.batch = pyglet.graphics.Batch()
        self.drawRoadmap = True
        self.policy = None
        self.env = None
        self.record = False
        self.reset_array()
        if checker is not None:
            vss=[]
            for x in range(0,wind_size[0],checker[0]):
                for y in range(0,wind_size[1],checker[1]):
                    if (x//checker[0]+y//checker[1])%2==0:
                        vss+=Viewer.get_box(checker[0], checker[1], lowerleft=(x,y), triangle=True)
            self.batch_checker=self.batch.add(len(vss)//2, pyglet.gl.GL_TRIANGLES, None, ('v2f', vss), ('c3B', tuple(checker[2])*(len(vss)//2)))
        else: self.batch_checker=None

    def reset_array(self):
        self.agent_pos_array = []
        self.batch_agent_list = []
        self.batch_agent_edge_list = []

        self.goal_pos_array = []
        self.batch_goal_list = []
        self.batch_goal_edge_list = []

        self.batch_obs_list = []
        self.batch_obs_edge_list = []

        self.waypoint_pos_array = []
        self.batch_waypoint_list = []
        self.batch_waypoint_edge_list = []
        
    def add_waypoint(self, pos, waypoint_size=10, color=[0, 0, 255]):
        self.waypoint_pos_array.append(list(pos))
        vss = Viewer.get_box(waypoint_size, waypoint_size)
        self.batch_waypoint_list.append(self.batch.add(len(vss) // 2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', vss),
                                                   ('c3B', tuple(color) * (len(vss) // 2))))
        self.batch_waypoint_edge_list.append(self.batch.add(len(vss) // 2, pyglet.gl.GL_LINE_LOOP, None, ('v2f', vss),
                                                        ('c3B', tuple([0, 0, 0]) * (len(vss) // 2))))

    def add_agent(self, pos, rad=10, color=[124,79,13]):
        self.agent_pos_array.append(list(pos))
        vss=Viewer.get_circle(rad)
        self.batch_agent_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', vss), ('c3B', tuple(color)*(len(vss)//2))))    
        vss=Viewer.get_circle(rad, fill=False)
        self.batch_agent_edge_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_LINE_STRIP, None, ('v2f', vss), ('c3B', tuple([0,0,0])*(len(vss)//2))))    

    def add_goal(self, pos, goal_size=10, color=[230,13,13]):
        self.goal_pos_array.append(list(pos))
        vss=Viewer.get_box(goal_size, goal_size)
        self.batch_goal_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', vss), ('c3B', tuple(color)*(len(vss)//2))))    
        self.batch_goal_edge_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_LINE_LOOP, None, ('v2f', vss), ('c3B', tuple([0,0,0])*(len(vss)//2))))    

    def add_obs(self, vss, color=[0,0,0]):
        self.batch_obs_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', vss), ('c3B', tuple(color)*(len(vss)//2))))    
        self.batch_obs_edge_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_LINE_STRIP, None, ('v2f', vss), ('c3B', tuple([0,0,0])*(len(vss)//2))))    

    def on_key_press(self, symbol, modifiers):
        if symbol==pyglet.window.key.R:
            if self.env is not None:
                self.env.reset()
            if self.policy is not None:
                self.policy.reset()
        elif symbol==pyglet.window.key.T:
            if self.policy is not None:
                self.policy.load_random_env('mazes_g100w700h700_Var10')
            elif self.env is not None:
                self.env.load_random_env('mazes_g100w700h700_Var10')
        elif symbol==pyglet.window.key.E:
            self.drawRoadmap=not self.drawRoadmap
        elif symbol==pyglet.window.key.Q:
            self.record=not self.record
            if self.record:
                self.animation_frms=[]
            elif len(self.animation_frms)>0:
                height,width,layers=self.animation_frms[0].shape
                out=cv2.VideoWriter('single2.avi',cv2.VideoWriter_fourcc(*'DIVX'),16,(self.width,self.height))
                for f in self.animation_frms:
                    out.write(f)
                out.release()
                if os.path.exists('frm.png'):
                    os.remove('frm.png')
        elif symbol==pyglet.window.key.ESCAPE:
            self.close()
    
    def on_draw(self):
        self.clear()
        pyglet.gl.glLineWidth(2)
        pyglet.gl.glPointSize(10)
        
        if self.batch_checker is not None:
            self.batch_checker.draw(pyglet.gl.GL_TRIANGLES)
        
        if self.drawRoadmap:
            if hasattr(self,'roadmap_edge'):
                self.roadmap_edge.draw(pyglet.gl.GL_LINES)
            if hasattr(self,'roadmap_edge_to_goal'):
                self.roadmap_edge_to_goal.draw(pyglet.gl.GL_LINES)
            if hasattr(self,'roadmap_vertex'):
                self.roadmap_vertex.draw(pyglet.gl.GL_POINTS)
            if hasattr(self,'medial_axis_vertex'):
                self.medial_axis_vertex.draw(pyglet.gl.GL_POINTS)
            if hasattr(self,'medial_axis_edge'):
                self.medial_axis_edge.draw(pyglet.gl.GL_LINES)

        for vertex_list,edge_list in zip(self.batch_obs_list,self.batch_obs_edge_list):
            vertex_list.draw(pyglet.gl.GL_TRIANGLE_FAN)
            edge_list.draw(pyglet.gl.GL_LINE_LOOP)
        
        for goal_pos,vertex_list,edge_list in zip(self.goal_pos_array,self.batch_goal_list,self.batch_goal_edge_list):
            pyglet.gl.glPushMatrix()
            pyglet.gl.glTranslatef(goal_pos[0],goal_pos[1],0)
            vertex_list.draw(pyglet.gl.GL_TRIANGLE_FAN)
            edge_list.draw(pyglet.gl.GL_LINE_LOOP)
            pyglet.gl.glPopMatrix()
        
        for agent_pos,vertex_list,edge_list in zip(self.agent_pos_array,self.batch_agent_list,self.batch_agent_edge_list):
            pyglet.gl.glPushMatrix()
            pyglet.gl.glTranslatef(agent_pos[0],agent_pos[1],0)
            vertex_list.draw(pyglet.gl.GL_TRIANGLE_FAN)
            edge_list.draw(pyglet.gl.GL_LINE_STRIP)
            pyglet.gl.glPopMatrix()

        for waypoint_pos,vertex_list,edge_list in zip(self.waypoint_pos_array,self.batch_waypoint_list,self.batch_waypoint_edge_list):
            pyglet.gl.glPushMatrix()
            pyglet.gl.glTranslatef(waypoint_pos[0],waypoint_pos[1],0)
            vertex_list.draw(pyglet.gl.GL_TRIANGLE_FAN)
            edge_list.draw(pyglet.gl.GL_LINE_LOOP)
            pyglet.gl.glPopMatrix()
            
        if self.record:
            import pyscreenshot as ImageGrab
            loc=self.get_location()
            bbox=(loc[0],loc[1],loc[0]+self.width,loc[1]+self.height)
            im=ImageGrab.grab(bbox).save('frm.png')
            self.animation_frms.append(cv2.imread('frm.png'))

        if hasattr(self,'sensor') and hasattr(self.sensor,'readings'):
            self.sensor.draw_sensor_reading()

    @staticmethod
    def get_circle(r, fill=True, RES=16):
        circle_list = [0.0, 0.0] if fill else []
        for i in range(RES+1):
            circle_list += [r*math.cos(math.pi*2*i/RES), r*math.sin(math.pi*2*i/RES)]
        return circle_list
        
    @staticmethod
    def get_box(x, y, ctr=None, lowerleft=None, triangle=False):
        if triangle:
            vss=[-x/2, -y/2, x/2, -y/2, x/2, y/2,   -x/2, -y/2, x/2, y/2, -x/2, y/2]
        else: vss=[-x/2, -y/2, x/2, -y/2, x/2, y/2, -x/2, y/2]
        if ctr is not None:
            for i in range(len(vss)//2):
                vss[i*2+0]+=ctr[0]
                vss[i*2+1]+=ctr[1]
        if lowerleft is not None:
            for i in range(len(vss)//2):
                vss[i*2+0]+=lowerleft[0]+x/2
                vss[i*2+1]+=lowerleft[1]+y/2
        return vss

    @staticmethod
    def get_box_ll(x, y, ctr=None, lowerleft=None, triangle=False):
        vss=Viewer.get_box(x, y, ctr, lowerleft, triangle)
        return [[vss[i],vss[i+1]] for i in range(0,len(vss),2)]

    def render(self):
        try:
            self.switch_to()
            self.dispatch_events()
            self.dispatch_event('on_draw')
            self.flip()
        except:
            self.close()
    
if __name__=='__main__':
    viewer=Viewer()
    viewer.add_agent(pos=(100,100))
    viewer.add_goal(pos=(700,700))
    viewer.add_obs(vss=Viewer.get_box(400, 400, lowerleft=(200,200)))
    pyglet.app.run()