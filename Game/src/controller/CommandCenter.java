package controller;

import java.util.ArrayList;

import exceptions.CannotTreatException;
import exceptions.IncompatibleTargetException;
import model.events.SOSListener;
import model.infrastructure.ResidentialBuilding;
import model.people.Citizen;
import model.units.Evacuator;
import model.units.FireTruck;
import model.units.GasControlUnit;
import model.units.Unit;
import simulation.Rescuable;
import simulation.Simulator;

public class CommandCenter implements SOSListener {

	private Simulator engine;
	private ArrayList<ResidentialBuilding> visibleBuildings;
	private ArrayList<Citizen> visibleCitizens;

	@SuppressWarnings("unused")
	private ArrayList<Unit> emergencyUnits;

	public CommandCenter() throws Exception {
		engine = new Simulator(this);
		visibleBuildings = new ArrayList<ResidentialBuilding>();
		visibleCitizens = new ArrayList<Citizen>();
		emergencyUnits = engine.getEmergencyUnits();

	}

	@Override
	public void receiveSOSCall(Rescuable r) {
		
		if (r instanceof ResidentialBuilding) {
			
			if (!visibleBuildings.contains(r))
				visibleBuildings.add((ResidentialBuilding) r);
			
		} else {
			
			if (!visibleCitizens.contains(r))
				visibleCitizens.add((Citizen) r);
		}

	}

	public Simulator getEngine() {
		return engine;
	}

	
	public ArrayList<ResidentialBuilding> getVisibleBuildings() {
		return visibleBuildings;
	}


	public ArrayList<Citizen> getVisibleCitizens() {
		return visibleCitizens;
	}

	

	public ArrayList<Unit> getEmergencyUnits() {
		return emergencyUnits;
	}
public void unitrespondlocation(Unit u , int x , int y ) throws Exception  {
	Rescuable t=null;
	//if(u instanceof Evacuator || u instanceof FireTruck || u instanceof GasControlUnit) {
	for(int i=0;i<visibleBuildings.size();i++) {
		ResidentialBuilding b = visibleBuildings.get(i);
		if(b.getLocation().getX()==x && b.getLocation().getY()==y) {
			t=b;
		}//else {
//			for(int j=0;j<visibleCitizens.size();j++) {
//				Citizen c= visibleCitizens.get(i);
//				if(c.getLocation().getX()==x&&c.getLocation().getY()==y) {
//					t=c;
//					
//				}
			//}//
		}
	
	
	//}
		for(int j=0;j<visibleCitizens.size();j++) {
			Citizen c = visibleCitizens.get(j);
			if(c.getLocation().getX()==x&&c.getLocation().getY()==y) {
				t=c;
			}else {
				for(int i=0;i<visibleBuildings.size();i++) {
					ResidentialBuilding b = visibleBuildings.get(i);
					if(b.getLocation().getX()==x && b.getLocation().getY()==y) {
						t=b;}}
			}
		}
	
	if(t==null) {
		return;
	}
	u.respond(t);
}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

}
