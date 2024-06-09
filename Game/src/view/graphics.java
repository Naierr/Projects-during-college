package view;
import java.awt.*;
import javax.imageio.*;
import model.units.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.net.NoRouteToHostException;
import java.util.ArrayList;

import controller.CommandCenter;
import exceptions.BuildingAlreadyCollapsedException;
import exceptions.CannotTreatException;
import exceptions.CitizenAlreadyDeadException;
import model.disasters.Disaster;
import model.infrastructure.ResidentialBuilding;
import model.people.Citizen;
import model.people.CitizenState;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.TitledBorder;

import controller.*; ; 
public class graphics extends JFrame implements ActionListener{
	// this class is a Jframe
	private CommandCenter base  ; 
	private JButton b1 ;
	private JButton play ; 
	private JButton Exit  ; 
	private JLabel gameover  ; 
	private JPanel gameGrid ; 
	private JPanel next ;
	private JPanel unitsRespond ;
	private JPanel log ; 
	private JPanel info ; 
	private JPanel core  ; 
	private int lastXCoordinate  ; 
	private int lastYCoordinate ; 
	private  JButton[][] world ;
	private JOptionPane p ; 
	private JTextArea area  ;
	private JTextField player_name  ; 
	private JLabel name ; 
	private JPasswordField password ; 
	private JLabel intro ; 
	private JButton f;
	private JButton e;
	private JButton d;
	private JButton g;
	private JButton a;
	private JPanel treating;
	private JPanel idle;
	private JPanel responding;
	private JTextField casualities;
	private JLabel esm;
private JTextArea information ; 
private JScrollPane S3 ; 
private String namee;
	//private JToggleButton tog  ; 
	//private JMenuBar menu ; 
	private JScrollPane SS ; 
	private JButton nextCycle;
	public graphics() throws Exception{
		base = new CommandCenter() ;
		world = new JButton[10][10] ;
		intro = new JLabel(new ImageIcon("kiko.jpg"));
		gameover = new JLabel(new ImageIcon("gameover2.png  ")) ;
		this.getContentPane().add(gameover);
		gameover.setVisible(false);
		gameover.setBounds(0, 0, 800, 600);
		intro.setBounds(0, 0, 1200, 700);
		
		this.intro.setLayout(null);
		this.getContentPane().add(intro);
		name = new JLabel("user_name");
		name.setBounds(20, 60, 100, 30);
		player_name = new JTextField("") ; 
		player_name.setBackground(Color.MAGENTA);
		player_name.setBounds(150, 60, 100, 30);
		play = new JButton("play") ; 
		play.addActionListener(this);
		play.setFont(new Font("Dialog",Font.BOLD,18));
		play.setBackground(Color.BLUE);
		play.setBounds(100, 150, 100,30); 
		this.intro.add(name) ;
		this.intro.add(play) ;
		this.intro.add(player_name) ;
		Exit = new JButton("Leave") ; 
		Exit.setBackground(Color.GREEN);
		//Exit.setBackground(Color.black);
		Exit.setBounds(900, 150, 100,30);
		Exit.setFont(new Font("Dialog",Font.BOLD,18));
        Exit.addActionListener(this);
        intro.add(Exit) ;
        core = new JPanel() ;
        core.setSize(1200, 700);
        Border blackline ,etched,raisedbevel,loweredbevel,empty;
        blackline=BorderFactory.createLineBorder(Color.BLACK);
        etched=BorderFactory.createEtchedBorder();
        raisedbevel=BorderFactory.createRaisedBevelBorder();
        loweredbevel=BorderFactory.createLoweredBevelBorder();
        empty=BorderFactory.createEmptyBorder();
        JLabel idleLabel=new JLabel("IDLE");
        idleLabel.setText("idle");
        Border border=BorderFactory.createLineBorder(Color.BLACK,10);
       this.core.setVisible(false);
        this.getContentPane().add(core) ; 
      core.setLayout(new BorderLayout());
        gameGrid = new JPanel() ;
        info = new JPanel() ;
         
        
        log = new JPanel() ;
        unitsRespond = new JPanel() ;
        next = new JPanel() ;
        core.add(next, BorderLayout.NORTH);
        core.add(gameGrid, BorderLayout.CENTER);
        core.add(unitsRespond, BorderLayout.WEST);
        core.add(log, BorderLayout.SOUTH);
        core.add(info, BorderLayout.EAST);
        this.getContentPane().add(core) ;
        gameGrid.setLayout(new GridLayout(10, 10));
        for(int i=0 ; i<10 ; i++) {
        	for(int j =0 ; j<10 ; j++) {
        		world[i][j] = new JButton() ; 
        		gameGrid.add(world[i][j]) ; 
        		world[i][j].addActionListener(this); 
        		world[i][j].setBorder(blackline);
        		//world[i][j].setBorder(etched);
        		world[i][j].setBackground(Color.white);
        		world[i][j].setToolTipText("" +i+"," +j);
        		// intial simulation information  
        		
        		
        	}}
        gameGrid.setPreferredSize(new Dimension(150,200));
        	
        	//this.icons();
        	unitsRespond.setLayout(new GridLayout(3, 1));
        	unitsRespond.setBackground(Color.BLUE);
        idle = new JPanel();
        unitsRespond.add(idle);
        idle.setBackground(Color.GREEN);
        idle.setLayout(new GridLayout(2,3));
        
        responding= new JPanel();
        unitsRespond.add(responding);
        responding.setLayout(new GridLayout(2,3));
        treating = new JPanel();
        unitsRespond.add(treating);
        treating.setLayout(new GridLayout(2,3));
        //units buttons
        responding.setBackground(Color.CYAN);
        treating.setBackground(Color.red);
        a= new JButton();
        a.addActionListener(this);
      //    a.setIcon(defaultIcon);   ambulance
        f= new JButton();
      //  f.setIcon(defaultIcon);   fireee
        f.addActionListener(this);
        d= new JButton();
      //  d.setIcon(defaultIcon); 
        d.addActionListener(this);
        g= new JButton();
        g.addActionListener(this);
 //       g.setIcon(defaultIcon);   gas   
        e= new JButton();
      //  e.setIcon(defaultIcon); evacuator
        e.addActionListener(this);
        
        idle.setPreferredSize(new Dimension(250, 400));
//        Border blackline ,etched,raisedbevel,loweredbevel,empty;
//        blackline=BorderFactory.createLineBorder(Color.BLACK);
//        etched=BorderFactory.createEtchedBorder();
//        raisedbevel=BorderFactory.createRaisedBevelBorder();
//        loweredbevel=BorderFactory.createLoweredBevelBorder();
//        empty=BorderFactory.createEmptyBorder();
       TitledBorder title1,title2,title3;
        title1=BorderFactory.createTitledBorder("IDLE");
        title2=BorderFactory.createTitledBorder("RESPONDING");
   title3=BorderFactory.createTitledBorder("TREATING");
        idle.setBorder(title1);
       // idle.setBorder(blackline);
        //idle.setBorder(etched);
        //idle.setBorder(raisedbevel);
       //idle.setBorder(loweredbevel);
      //  idle.setBorder(empty);
        responding.setBorder(title2);
        treating.setBorder(title3);
        idle.add(a); 
        a.setIcon(new ImageIcon("ambulance.jpg "));
        idle.add(f);
        f.setIcon(new ImageIcon("firetruck.png "));
        idle.add(d);
        d.setIcon(new ImageIcon("disease.jpg"));
        idle.add(g);
        g.setIcon(new ImageIcon("gas.jpg "));
        idle.add(e);
        e.setIcon(new ImageIcon("halettwar2geem.jpg "));
        area = new JTextArea() ;
        area.setBackground(Color.lightGray);
        area.setEditable(false);
      //  area.setLineWrap(true);
		//area.setWrapStyleWord(true);
		 
     this.logHelper();// set text  
     SS = new JScrollPane(area,JScrollPane.VERTICAL_SCROLLBAR_ALWAYS,JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS) ;
     SS.setPreferredSize(new Dimension(100, 100));
     log.add(SS) ;
     log.setLayout(new GridLayout(1, 1));
	
	 info.setLayout(new GridLayout(1, 1)); 
	 information = new JTextArea()  ; 
	 information.setEditable(false);
	 information.setLineWrap(true);
	 information.setBackground(Color.LIGHT_GRAY);
	S3 = new JScrollPane(information,JScrollPane.VERTICAL_SCROLLBAR_ALWAYS,JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS) ;
	info.add(S3) ; 
	S3.setPreferredSize(new Dimension(200, 600));

        nextCycle=new JButton("Next Cycle");
        nextCycle.addActionListener(this);
        next.setLayout(new GridLayout(1,3));
        next.add(nextCycle);
        namee=player_name.getText();
        esm=new JLabel(" Current Cycle   : " + ""  +this.base.getEngine().getCurrentCycle());
        next.add(esm);
        casualities=new JTextField("deaths   :   0");
        next.add(casualities);
        casualities.setEditable(false);
        this.icons();
	this.setSize(1200, 700) ;
	this.setState(Frame.ICONIFIED);
	this.setResizable(false);
	this.validate(); 
	this.repaint(); 
	

		this.setVisible(true);
	}
	public static void main(String[] args) throws Exception {
		graphics c = new graphics() ;
	}
	public void actionPerformed(ActionEvent e) {
	if(e.getSource() == Exit) {
		System.exit(0);
	}if ( e.getSource()==play) {
		intro.setVisible(false);
		//this.getContentPane().setLayout(new BorderLayout());
		core.setVisible(true);	}
	 for(int i =0 ; i<10;i++) {
		 for(int j = 0; j<10 ; j++) {
			 if(e.getSource()==world[i][j]) {
				 lastXCoordinate = i ; 
				 lastYCoordinate = j ;
				 this.buttonInfo(i, j);
				
			 }
		 }
	 }
	 if(e.getSource()==a) { /// ambulance
		 for(int i =0 ; i<this.base.getEmergencyUnits().size();i++) {
			 Unit u = this.base.getEmergencyUnits().get(i) ; 
			 if( u instanceof Ambulance) {
				try {
					base.unitrespondlocation(u, lastXCoordinate, lastYCoordinate);
				}
				catch (Exception e1) {
					JOptionPane.showMessageDialog(this, e1.getMessage(),
							"error", JOptionPane.ERROR_MESSAGE);
				}
			 }
		 }
	 }  if(e.getSource()==d) { /// disease
		 for(int i =0 ; i<this.base.getEmergencyUnits().size();i++) {
			 Unit u = this.base.getEmergencyUnits().get(i) ; 
			 if( u instanceof DiseaseControlUnit) {
				try {
					base.unitrespondlocation(u, lastXCoordinate, lastYCoordinate);
				} catch (Exception e1) {
					JOptionPane.showMessageDialog(this, e1.getMessage(),
							"error", JOptionPane.ERROR_MESSAGE);
				}
			 }
		 }}  if( e.getSource()==f) {
			 for(int i =0 ; i<this.base.getEmergencyUnits().size();i++) {
				 Unit u = this.base.getEmergencyUnits().get(i) ; 
				 if( u instanceof FireTruck) {
					try {
						base.unitrespondlocation(u, lastXCoordinate, lastYCoordinate);
					} catch (Exception e2) {
						JOptionPane.showMessageDialog(this, e2.getMessage(),
								"error404", JOptionPane.ERROR_MESSAGE);
					}
				 }}
			 
		 } if( e.getSource()==g) {
			 for(int i =0 ; i<this.base.getEmergencyUnits().size();i++) {
				 Unit u = this.base.getEmergencyUnits().get(i) ; 
				 if( u instanceof GasControlUnit) {
					try {
						base.unitrespondlocation(u, lastXCoordinate, lastYCoordinate);
					} catch (Exception e3) {
						JOptionPane.showMessageDialog(this, e3.getMessage(),
								"error404", JOptionPane.ERROR_MESSAGE);
					}
				 }
			 }
		 } if( e.getSource()==this.e) {
			 for(int i =0 ; i<this.base.getEmergencyUnits().size();i++) {
				 Unit u = this.base.getEmergencyUnits().get(i) ; 
				 if( u instanceof Evacuator) {
					try {
						base.unitrespondlocation(u, lastXCoordinate, lastYCoordinate);
					} catch (Exception e4) {
						JOptionPane.showMessageDialog(this, e4.getMessage(),
								"error404", JOptionPane.ERROR_MESSAGE);
					}
				 }
			 
		 }}  if(e.getSource()== nextCycle) {
			 try {
				 
				this.base.getEngine().nextCycle();
				this.logHelper();
				this.updateUnitStates();  
				this.icons();	
				esm.setText(" Current Cycle   : " + ""  +this.base.getEngine().getCurrentCycle());
				casualities.setText(" Deaths"+ "" + this.base.getEngine().calculateCasualties());
				if(this.base.getEngine().checkGameOver()) {
					gameover.setVisible(true);

					JOptionPane.showMessageDialog(this," GAME OVER YOU LOST TO THE MIGHTY MOJO JOJO", " Deaths = "+this.base.getEngine().calculateCasualties() ,JOptionPane.ERROR_MESSAGE);
									
					System.exit(0);
				}
				
				 
				 
			} catch (BuildingAlreadyCollapsedException e5) {
				JOptionPane.showMessageDialog(this, e5.getMessage(), "STahhhpp it is already DEAD", JOptionPane.ERROR_MESSAGE);
			} catch (CitizenAlreadyDeadException e6) {
				JOptionPane.showMessageDialog(this, e6.getMessage(), " Staahpp you sadistic", JOptionPane.ERROR_MESSAGE);
			}
			 
			 
			 	 
		 }
		 }
	
public void icons()	{
	world[0][0].setIcon(new ImageIcon("commandd.jpg  "));
	for(int i = 0 ; i<base.getEngine().getCitizens().size();i++) {
		Citizen c=base.getEngine().getCitizens().get(i);
		int xx = c.getLocation().getX() ; 
		int yy =   c.getLocation().getY() ;
		if(c.getState()==CitizenState.DECEASED) {
			 world[xx][yy].setIcon(new ImageIcon("deadd.jpg"));
		}
		else {
			if(c.getState()==CitizenState.RESCUED) {
				world[xx][yy].setIcon(new ImageIcon("safecitizen.jpg"));
			}
			if(c.getState()==CitizenState.SAFE) {
				 world[xx][yy].setIcon(new ImageIcon("safecitizen.jpg"));
			}
			else {
				if(c.getState()==CitizenState.IN_TROUBLE) {
					 world[xx][yy].setIcon(new ImageIcon("injury.jpg")) ;
				
				}
			}
		}}
	for(int i =0 ; i< base.getEngine().getBuildings().size();i++) {
		ResidentialBuilding r = base.getEngine().getBuildings().get(i) ; 
		int xx = r.getLocation().getX() ; 
		int yy =   r.getLocation().getY() ; // location ely han update be el buttons 
		if(r.getOccupants().size()==0 && r.getDisaster()==null &&r.getStructuralIntegrity()!=0 ) {
			 world[xx][yy].setIcon(new ImageIcon("safeBuilding.jpg")) ;
		}else if (r.getOccupants().size()>0 && r.getDisaster()!=null  )  {
			 world[xx][yy].setIcon(new ImageIcon("citizeninbuilding.jpg")) ;
			 if(r.getDisaster().isActive()==false) {
				 world[xx][yy].setIcon(new ImageIcon("safeBuilding.jpg")) ;
			 }
		} else if (r.getOccupants().size()>=0 && r.getDisaster()==null &&r.getStructuralIntegrity()!=0) {
			 world[xx][yy].setIcon(new ImageIcon("safeBuilding.jpg")) ;
		}
		if(r.getStructuralIntegrity()==0) {
			world[xx][yy].setIcon(new ImageIcon("collapse.jpg")) ;
		}
		}
	
	}
	public  void logHelper() {
		String s="";
		String gg = "active" ;
		for(int i=0;i<base.getEngine().getExecutedDisasters().size();i++) {
		
			Disaster p=(Disaster)base.getEngine().getExecutedDisasters().get(i);
			
			if(p.isActive()==false) {
				 gg = " inactive" ;
			}
			s+= "Disaster status: " +gg+ " ,  " + "DisasterType : "+ p.getType()+ " , " + "StartCycle :  " +"  " +p.getStartCycle() ;
			if(p.getTarget() instanceof Citizen ) {
				Citizen y=(Citizen)p.getTarget();
				s+= " name : " + y.getName() +" , " + " CitizenLocation" + " "+ y.getLocation().getX()+ " , " + y.getLocation().getY();
			}
			if(p.getTarget() instanceof ResidentialBuilding) {
				ResidentialBuilding m= (ResidentialBuilding)p.getTarget();
				s+= " Building" + " location " + "  " + m.getLocation().getX() +" , " + m.getLocation().getY();
			}
			s+="\n";
		}
    area.setText(s );
     /// reconstruct the scroll dimesnsion       
    
	}
    public void buttonInfo(int x , int y ) {
    	String s = "" ; 
    	for(int i =0 ; i<base.getEngine().getBuildings().size();i++) {
    		ResidentialBuilding r = base.getEngine().getBuildings().get(i) ; 
    		if(r.getLocation().getX()==x && r.getLocation().getY()==y) {
    			
				s+= " Building" + " and its location is " + " " + r.getLocation().getX() + r.getLocation().getY() ; 
			s+="\n";
			s+= " StrucutralIntegrity : " + r.getStructuralIntegrity()  ; 
			s+="\n";
			s+= " FireDamage : " + r.getFireDamage()  ; 
			s+="\n";
			s+= " GasLevel : " + r.getGasLevel()  ; 
			s+="\n";
			s+= " FoundationDamage : " + r.getFoundationDamage()  ; 
			s+="\n";
			if( r.getOccupants().size()!=0) {
			s+= " Occupants : " + r.getOccupants().size()  ; 
			s+="\n"; 
			 s+= " Occupants info " +  "\n" + infoCitizen(x, y) ; }
			s+= " Units"+"\n" + this.unitinfo(x, y) ;
			
    		}
    		s+= infoCitizen(x, y) ; 
    	}
    	
    	information.setText(s);//see that  	
    }
	public String infoCitizen(int x , int y ) {	
		String s="";
		for(int i=0;i<base.getEngine().getCitizens().size();i++) {
	            Citizen c= (Citizen)base.getEngine().getCitizens().get(i);
				if(c.getLocation().getX()==x&&c.getLocation().getY()==y) {
				s+="the Citizen location is"+" "+c.getLocation().getX()+c.getLocation().getY() + "\n" ;  ;
				s+="name:"+" "+c.getName() +"  "+"age"+" "+c.getAge()+" "+"national ID"+" "+c.getNationalID();
				s+="\n";
				s+="hp :"+ " "+c.getHp();
				s+="\n";
				s+="blood loss :"+" "+c.getBloodLoss();
				s+="\n";
				s+="toxicity :"+" "+c.getToxicity();
				s+="\n";
				s+="citizen state:"+" "+c.getState();
				s+="\n";
				if(c.getDisaster()!=null) {
					s+="disaster affecting citizen:"+" "+c.getDisaster().getType();
				}
				
			}
		}return s;
	}
		public String  unitinfo(int x , int y  ) {
			String s="";
			for(int i=0;i<base.getEmergencyUnits().size();i++) {
		            Unit u= (Unit) base.getEngine().getEmergencyUnits().get(i);
					if(u.getLocation().getX()==x&&u.getLocation().getY()==y) {
					s+="the location is"+" "+u.getLocation().getX() + u.getLocation().getY();
					s+="\n";
					s+="type:"+" "+u.getUnitType() +"  "+"ID"+" "+u.getUnitID()+" "+"steps"+" "+u.getStepsPerCycle()  ;
					s+="\n";
					s+="state"+" "+u.getState()  ;
					if(u.getTarget() instanceof Citizen ) {
						Citizen c=(Citizen)u.getTarget();
						s+= "target " + c.getName() + " and its location is " + " " + c.getLocation().getX() + c.getLocation().getY();
					}
					if(u.getTarget() instanceof ResidentialBuilding) {
						ResidentialBuilding m= (ResidentialBuilding)u.getTarget();
						s+= " Building" + " and its location is " + " " + m.getLocation().getX() + m.getLocation().getY();
					}
					s+="\n";
					if( u instanceof Evacuator) {
						Evacuator e = (Evacuator) u ; 
						s+= " occupants" + e.getPassengers().size()  ; 
						s+=           "\n" ; 
						for(int k =0 ; k<e.getPassengers().size();k++) {
							s+= this.infoCitizen(x, y) ; 
						}
				}				
					}}
			
			return s ; 	
		}
		public void updateUnitStates () {
			idle.removeAll();
			treating.removeAll();
			responding.removeAll();
			for(int i =0 ; i<this.base.getEmergencyUnits().size();i++) {
				Unit u = this.base.getEmergencyUnits().get(i) ;
				if( u instanceof Ambulance) {
					if(u.getState()== UnitState.IDLE) {
						idle.add(a) ; 
						
					}else if (u.getState()==UnitState.RESPONDING) {
						responding.add(a) ;
					}else {
						treating.add(a) ; 
					}
				}else 	if( u instanceof FireTruck) {
					if(u.getState()== UnitState.IDLE) {
						idle.add(f) ; 
						
					}else if (u.getState()==UnitState.RESPONDING) {
						responding.add(f) ;
					}else {
						treating.add(f) ; 
					}	
				} else 	if( u instanceof DiseaseControlUnit) {
					if(u.getState()== UnitState.IDLE) {
						idle.add(d) ; 
						
					}else if (u.getState()==UnitState.RESPONDING) {
						responding.add(d) ;
					}else {
						treating.add(d) ; 
					}	
				} else 	if( u instanceof GasControlUnit) {
					if(u.getState()== UnitState.IDLE) {
						idle.add(g) ; 
						
					}else if (u.getState()==UnitState.RESPONDING) {
						responding.add(g) ;
					}else {
						treating.add(g) ; 
					}	
				} else if ( u instanceof Evacuator) {
					if(u.getState()== UnitState.IDLE) {
						idle.add(e) ; 
						
					}else if (u.getState()==UnitState.RESPONDING) {
						responding.add(e) ;
					}else {
						treating.add(e) ; 
					}	
				}
			}
			
			unitsRespond.validate();
			unitsRespond.repaint(); 
			
		}
}
