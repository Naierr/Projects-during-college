package exceptions;
import model.units.*;
import simulation.*;
public class CannotTreatException extends UnitException {

	public CannotTreatException(Unit unit,Rescuable target) {
		super(unit,target);
	}
	public CannotTreatException(Unit unit,Rescuable target,String message) {
		super(unit,target,message);
	}
}
