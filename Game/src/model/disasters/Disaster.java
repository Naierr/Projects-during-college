package model.disasters;

import exceptions.BuildingAlreadyCollapsedException;
import exceptions.CitizenAlreadyDeadException;
import model.infrastructure.ResidentialBuilding;
import model.people.Citizen;
import model.people.CitizenState;
import simulation.Rescuable;
import simulation.Simulatable;

public abstract class Disaster implements Simulatable{
	private int startCycle;
	private Rescuable target;
	private boolean active;
	private String type  ;
	public Disaster(int startCycle, Rescuable target) {
		this.startCycle = startCycle;
		this.target = target;
	}
	public boolean isActive() {
		return active;
	}
	public void setActive(boolean active) {
		this.active = active;
	}
	public int getStartCycle() {
		return startCycle;
	}
	public Rescuable getTarget() {
		return target;
	}
	public void strike() throws BuildingAlreadyCollapsedException , CitizenAlreadyDeadException
	{
		if(this.getTarget() instanceof Citizen) {
			Citizen c = (Citizen) this.getTarget();
			if(c.getState()==CitizenState.DECEASED ) {
				throw new CitizenAlreadyDeadException(this , "citizen is already dead");
			}
		}else {
			ResidentialBuilding b= (ResidentialBuilding)this.getTarget();
			if(b.getStructuralIntegrity()==0) {
				throw new BuildingAlreadyCollapsedException(this, " building already collapsed") ; 
				
			}
		}
		target.struckBy(this);
		active=true;
	}
	public String getType() {
		return type;
	}
	public void setType(String type) {
		this.type = type;
	}
}
