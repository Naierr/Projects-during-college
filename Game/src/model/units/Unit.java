package model.units;


import exceptions.CannotTreatException;
import exceptions.IncompatibleTargetException;
import model.disasters.Collapse;
import model.disasters.Disaster;
import model.events.SOSResponder;
import model.events.WorldListener;
import model.infrastructure.ResidentialBuilding;
import model.people.Citizen;
import model.people.CitizenState;
import simulation.Address;
import simulation.Rescuable;
import simulation.Simulatable;

public abstract class Unit implements Simulatable, SOSResponder {
	private String unitID;
	private UnitState state;
	private Address location;
	private Rescuable target;
	private int distanceToTarget;
	private int stepsPerCycle;
	private WorldListener worldListener;
	private int maxStepsPerCycle=20;
	private String unitType  ;
	public void setStepsPerCycle(int stepsPerCycle) {
		if(stepsPerCycle>=maxStepsPerCycle) {
			stepsPerCycle=maxStepsPerCycle;
		}
		this.stepsPerCycle = stepsPerCycle;
	}

	public void setUnitType(String unitType) {
		this.unitType = unitType;
	}

	public int getDistanceToTarget() {
		return distanceToTarget;
	}

	public String getUnitType() {
		return unitType;
	}

	public void setUnitID(String unitID) {
		this.unitID = unitID;
	}

	public void setTarget(Rescuable target) {
		this.target = target;
	}

	public void setMaxStepsPerCycle(int maxStepsPerCycle) {
		this.maxStepsPerCycle = maxStepsPerCycle;
	}

	public int getMaxStepsPerCycle() {
		return maxStepsPerCycle;
	}

	public Unit(String unitID, Address location, int stepsPerCycle,
			WorldListener worldListener) {
		this.unitID = unitID;
		this.location = location;
		this.stepsPerCycle = stepsPerCycle;
		this.state = UnitState.IDLE;
		this.worldListener = worldListener;
	}

	public void setWorldListener(WorldListener listener) {
		this.worldListener = listener;
	}

	public WorldListener getWorldListener() {
		return worldListener;
	}

	public UnitState getState() {
		return state;
	}

	public void setState(UnitState state) {
		this.state = state;
	}

	public Address getLocation() {
		return location;
	}

	public void setLocation(Address location) {
		this.location = location;
	}

	public String getUnitID() {
		return unitID;
	}

	public Rescuable getTarget() {
		return target;
	}

	public int getStepsPerCycle() {
		return stepsPerCycle;
	}

	public void setDistanceToTarget(int distanceToTarget) {
		this.distanceToTarget = distanceToTarget;
		
	}

	@Override
	public void respond(Rescuable r) throws IncompatibleTargetException,CannotTreatException  {

	
		if(this instanceof FireUnit || this instanceof Evacuator) {
			if(!(r instanceof ResidentialBuilding)) {
				throw new IncompatibleTargetException(this,r, "Unit should be assigned to a compatible target");
		
		}}
		if((!canTreat(r)) || r.getDisaster()==null ) {
			throw new CannotTreatException(this,r,"The target is already safe");
}
		
		
		if (target != null && state == UnitState.TREATING) {
			reactivateDisaster();}

		finishRespond(r);
	}
	
		

	public void reactivateDisaster() {
		Disaster curr = target.getDisaster();
		curr.setActive(true);
	}

	public void finishRespond(Rescuable r) {
		target = r;
		state = UnitState.RESPONDING;
		Address t = r.getLocation();
		distanceToTarget = Math.abs(t.getX() - location.getX())
				+ Math.abs(t.getY() - location.getY());

	}

	public abstract void treat();

	public boolean canTreat(Rescuable r){
		if(r ==null)
			return false;
		if(this instanceof Ambulance && r instanceof Citizen) {
			Citizen v=(Citizen) r;
			if(v.getBloodLoss()==0)
				return false;
			
				}
		if(this instanceof DiseaseControlUnit && r instanceof Citizen) {
			Citizen v=(Citizen) r;
			if(v.getToxicity()==0)
				return false;
		}
		if(this instanceof FireTruck && r instanceof ResidentialBuilding) {
			ResidentialBuilding h= (ResidentialBuilding) r;
			if(h.getFireDamage()==0) {
				return false;
			}
			
		}
		if(this instanceof GasControlUnit && r instanceof ResidentialBuilding) {
			ResidentialBuilding h= (ResidentialBuilding) r;
			if(h.getGasLevel()==0) {
				return false;
			}
		}
		if(this instanceof Evacuator && r instanceof ResidentialBuilding) {
			ResidentialBuilding h= (ResidentialBuilding) r;
			if(h.getFoundationDamage()==0 ) {
				return false;
			}
		}
		return true;
		}
	public void cycleStep() {
		if (state == UnitState.IDLE)
			return;
		if (distanceToTarget > 0) {
			distanceToTarget = distanceToTarget - stepsPerCycle;
			if (distanceToTarget <= 0) {
				distanceToTarget = 0;
				Address t = target.getLocation();
				worldListener.assignAddress(this, t.getX(), t.getY());
			}
		} else {
			state = UnitState.TREATING;
			treat();
		}
	}

	public void jobsDone() {
		target = null;
		if(this instanceof FireTruck) {
			this.setStepsPerCycle(this.getStepsPerCycle()+3);	
		}
		else {
		this.setStepsPerCycle(this.getStepsPerCycle()+2);
		}
		state = UnitState.IDLE;

	}
}
