package tests;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import org.junit.Test;
import model.disasters.Collapse;
import model.disasters.Disaster;
import model.disasters.Fire;
import model.disasters.GasLeak;
import model.disasters.Infection;
import model.disasters.Injury;
import model.events.SOSListener;
import model.events.WorldListener;
import model.infrastructure.ResidentialBuilding;
import model.people.Citizen;
import model.units.Ambulance;
import model.units.DiseaseControlUnit;
import model.units.Evacuator;
import model.units.FireTruck;
import model.units.GasControlUnit;
import model.units.MedicalUnit;
import model.units.Unit;
import model.units.UnitState;
import simulation.Address;
import simulation.Rescuable;
import simulation.Simulator;

//,"unused"
@SuppressWarnings({ "rawtypes", "unchecked" })
public class Quiz2_V1 {
	Address testAddress1 = new Address(0, 0);
	Address testAddress2 = new Address(1, 1);
	Address testAddress3 = new Address(2, 3);
	Address testAddress4 = new Address(4, 4);

	String id1 = "1";
	String id2 = "2";
	String id3 = "3";
	String id4 = "4";

	String name1 = "test1";
	String name2 = "test2";
	String name3 = "test3";
	String name4 = "test4";

	int age1 = 24;
	int age2 = 25;
	int age3 = 26;
	int age4 = 27;

	Citizen testCitizen1 = new Citizen(testAddress1, id1, name1, age1, null);
	Citizen testCitizen2 = new Citizen(testAddress2, id2, name2, age2, null);
	Citizen testCitizen3 = new Citizen(testAddress3, id3, name3, age3, null);
	Citizen testCitizen4 = new Citizen(testAddress4, id4, name4, age4, null);

	ResidentialBuilding testBuilding1 = new ResidentialBuilding(testAddress1);
	ResidentialBuilding testBuilding2 = new ResidentialBuilding(testAddress2);
	ResidentialBuilding testBuilding3 = new ResidentialBuilding(testAddress3);
	ResidentialBuilding testBuilding4 = new ResidentialBuilding(testAddress4);

	Ambulance testAmbulance = new Ambulance(id1, testAddress1, 1, null);
	Evacuator testEvacutor = new Evacuator(id2, testAddress1, 1, null, 5);
	FireTruck testFireTruck = new FireTruck(id3, testAddress1, 1, null);
	DiseaseControlUnit testDiseaseContorlUnit = new DiseaseControlUnit(id4,
			testAddress1, 1, null);
	GasControlUnit testGasContorlUnit = new GasControlUnit("5", testAddress1,
			1, null);

	Collapse testCollapse = new Collapse(1, testBuilding1);
	Fire testFire = new Fire(3, testBuilding1);
	GasLeak testGasLeak = new GasLeak(3, testBuilding1);
	Infection testiInfection = new Infection(1, testCitizen1);
	Injury testinjInjury = new Injury(2, testCitizen2);

	static final String buildingPath = "model.infrastructure.ResidentialBuilding";
	static final String disasterPath = "model.disasters.Disaster";
	static final String sosListenerPath = "model.events.SOSListener";

	static String simulatorPath = "simulation.Simulator";
	static String addressPath = "simulation.Address";
	static String rescuablePath = "simulation.Rescuable";
	static String simulatablePath = "simulation.Simulatable";
	static String residentialBuildingPath = "model.infrastructure.ResidentialBuilding";
	static String citizenStatePath = "model.people.CitizenState";
	static String unitStatePath = "model.units.UnitState";
	static String citizenPath = "model.people.Citizen";
	static String unitPath = "model.units.Unit";
	static String policeUnitPath = "model.units.PoliceUnit";
	static String fireUnitPath = "model.units.FireUnit";
	static String medicalUnitPath = "model.units.MedicalUnit";
	static String evacuatorPath = "model.units.Evacuator";
	static String fireTruckPath = "model.units.FireTruck";
	static String gasControlUnitPath = "model.units.GasControlUnit";
	static String ambulancePath = "model.units.Ambulance";
	static String diseaseControlUnitPath = "model.units.DiseaseControlUnit";
	static String collapsePath = "model.disasters.Collapse";
	static String firePath = "model.disasters.Fire";
	static String gasLeakPath = "model.disasters.GasLeak";
	static String infectionPath = "model.disasters.Infection";
	static String injuryPath = "model.disasters.Injury";
	static String commandCenterPath = "controller.CommandCenter";
	static String worldListenerPath = "model.events.WorldListener";
	static String sosResponderPath = "model.events.SOSResponder";

	public static boolean treatCalled = false;
	public static boolean healCalled = false;
	@SuppressWarnings("unused")
	private static boolean struckByCalled = false;

	Infection testInfection = new Infection(1, testCitizen1);
	Injury testInjury = new Injury(2, testCitizen2);

	HashMap<String, Integer> count;
	SOSListener sos = new SOSListener() {

		@Override
		public void receiveSOSCall(Rescuable r) {

		}
	};

	@Test(timeout = 1000)
	public void testInstanceVariableUnitMaxStepsPerCycle() throws Exception {
		testInstanceVariableIsPresent(Class.forName(unitPath),
				"maxStepsPerCycle", true);
		testInstanceVariableIsPrivate(Class.forName(unitPath),
				"maxStepsPerCycle");
	}

	@Test(timeout = 100)
	public void testInstanceVariableUnitMaxStepsPerCycleNotInSubClasses()
			throws Exception {
		testInstanceVariableIsPresent(Class.forName(ambulancePath),
				"maxStepsPerCycle", false);
		testInstanceVariableIsPresent(Class.forName(diseaseControlUnitPath),
				"maxStepsPerCycle", false);
		testInstanceVariableIsPresent(Class.forName(evacuatorPath),
				"maxStepsPerCycle", false);
		testInstanceVariableIsPresent(Class.forName(fireTruckPath),
				"maxStepsPerCycle", false);
		testInstanceVariableIsPresent(Class.forName(fireUnitPath),
				"maxStepsPerCycle", false);
		testInstanceVariableIsPresent(Class.forName(gasControlUnitPath),
				"maxStepsPerCycle", false);
		testInstanceVariableIsPresent(Class.forName(medicalUnitPath),
				"maxStepsPerCycle", false);
		testInstanceVariableIsPresent(Class.forName(policeUnitPath),
				"maxStepsPerCycle", false);

	}

	@Test(timeout = 1000)
	public void testInstanceVariableUnitMaxStepsPerCycleGetterAndSetter()
			throws Exception {
		testGetterMethodExistsInClass(Class.forName(unitPath),
				"getMaxStepsPerCycle", int.class, true);
		testSetterMethodExistsInClass(Class.forName(unitPath),
				"setMaxStepsPerCycle", int.class, false);
	}

	@Test(timeout = 1000)
	public void testInstanceVariableUnitMaxStepsPerCycleGetterLogic()
			throws Exception {
		String String9 = "dds";
		int int0 = 0;
		int int3 = 3;
		Object address0 = Class.forName(addressPath)
				.getConstructor(int.class, int.class).newInstance(int0, int3);
		int int5 = 5;
		int int9 = 9;
	
		Object evacuator2 = Class
				.forName(evacuatorPath)
				.getConstructor(String.class, Class.forName(addressPath),
						int.class, WorldListener.class, int.class)
				.newInstance(String9, address0, int5, null, int9);

		testGetterLogic(evacuator2, "maxStepsPerCycle", 20);

		String9 = "amb";
		int0 = 1;
		int3 = 3;
		address0 = Class.forName(addressPath)
				.getConstructor(int.class, int.class).newInstance(int0, int3);
	
		Object ambulance1 = Class
				.forName(ambulancePath)
				.getConstructor(String.class, Class.forName(addressPath),
						int.class, WorldListener.class)
				.newInstance(String9, address0, int9, null);
		testGetterLogic(ambulance1, "maxStepsPerCycle", 20);

		String9 = "dcu";
		int0 = 2;
		int3 = 3;
		address0 = Class.forName(addressPath)
				.getConstructor(int.class, int.class).newInstance(int0, int3);
		Object diseaseControlUnit1 = Class
				.forName(diseaseControlUnitPath)
				.getConstructor(String.class, Class.forName(addressPath),
						int.class, WorldListener.class)
				.newInstance(String9, address0, int9, null);
		testGetterLogic(diseaseControlUnit1, "maxStepsPerCycle", 20);

		String9 = "ft";
		int0 = 3;
		int3 = 3;
		address0 = Class.forName(addressPath)
				.getConstructor(int.class, int.class).newInstance(int0, int3);
		Object fireTruck1 = Class
				.forName(fireTruckPath)
				.getConstructor(String.class, Class.forName(addressPath),
						int.class, WorldListener.class)
				.newInstance(String9, address0, int5, null);
		testGetterLogic(fireTruck1, "maxStepsPerCycle", 20);

		String9 = "gcu";
		int0 = 4;
		int3 = 3;
		address0 = Class.forName(addressPath)
				.getConstructor(int.class, int.class).newInstance(int0, int3);
		Object gasControlUnit1 = Class
				.forName(gasControlUnitPath)
				.getConstructor(String.class, Class.forName(addressPath),
						int.class, WorldListener.class)
				.newInstance(String9, address0, int9, null);
		testGetterLogic(gasControlUnit1, "maxStepsPerCycle", 20);
	}

	@Test(timeout = 1000)
	public void testInstanceVariableUnitStepsPerCycleSetter() throws Exception {
		testSetterMethodExistsInClass(Class.forName(unitPath),
				"setStepsPerCycle", int.class, true);
	}

	@Test(timeout = 1000)
	public void testConstructorInitializationUnitSetMaxStepsPerCycleCorrectly()
			throws Exception {
		String String0 = "io";
		int int7 = 7;
		int int4 = 4;
		Object address2 = Class.forName(addressPath)
				.getConstructor(int.class, int.class).newInstance(int7, int4);
		int int9 = 9;
		Object ambulance1 = Class
				.forName(ambulancePath)
				.getConstructor(String.class, Class.forName(addressPath),
						int.class, Class.forName(worldListenerPath))
				.newInstance(String0, address2, int9, null);
		String[] names = { "maxStepsPerCycle" };
		Object[] values = { 20 };
		testConstructorInitialization(ambulance1, names, values);

		int int0 = 0;
		int int3 = 3;
		String String9 = "dds";
		Object address0 = Class.forName(addressPath)
				.getConstructor(int.class, int.class).newInstance(int0, int3);
		int int5 = 5;
		int9 = 9;
		Object evacuator2 = Class
				.forName(evacuatorPath)
				.getConstructor(String.class, Class.forName(addressPath),
						int.class, Class.forName(worldListenerPath), int.class)
				.newInstance(String9, address0, int5, null, int9);
		testConstructorInitialization(evacuator2, names, values);

		String9 = "dcu";
		int0 = 2;
		int3 = 3;
		address0 = Class.forName(addressPath)
				.getConstructor(int.class, int.class).newInstance(int0, int3);
		Object diseaseControlUnit1 = Class
				.forName(diseaseControlUnitPath)
				.getConstructor(String.class, Class.forName(addressPath),
						int.class, Class.forName(worldListenerPath))
				.newInstance(String9, address0, int9, null);
		testConstructorInitialization(diseaseControlUnit1, names, values);

		String9 = "ft";
		int0 = 3;
		int3 = 3;
		address0 = Class.forName(addressPath)
				.getConstructor(int.class, int.class).newInstance(int0, int3);
		Object fireTruck1 = Class
				.forName(fireTruckPath)
				.getConstructor(String.class, Class.forName(addressPath),
						int.class, Class.forName(worldListenerPath))
				.newInstance(String9, address0, int5, null);
		testConstructorInitialization(fireTruck1, names, values);

		String9 = "gcu";
		int0 = 4;
		int3 = 3;
		address0 = Class.forName(addressPath)
				.getConstructor(int.class, int.class).newInstance(int0, int3);
		Object gasControlUnit1 = Class
				.forName(gasControlUnitPath)
				.getConstructor(String.class, Class.forName(addressPath),
						int.class, Class.forName(worldListenerPath))
				.newInstance(String9, address0, int9, null);
		testConstructorInitialization(gasControlUnit1, names, values);

	}
	
	@Test(timeout = 3000)
	public void testUnitsIncreasesStepsPerCycleAfterMission() throws Exception {
		// Unit 1: Ambulance
		Simulator s = new Simulator(sos);
		Ambulance u = new Ambulance("ambulance1", new Address(3, 4), 3, null);
		u.setWorldListener(s);
		int oldStepsPerCycle = u.getStepsPerCycle();
		Citizen c1 = new Citizen(new Address(3, 9), "1", "citizen1", 15, null);
		Disaster d = new Injury(3, c1);
		dshelper(d);
		c1.setBloodLoss(50);
		unitRespond(u, c1, 5);

		u.setState(UnitState.TREATING);

		for (int i = 0; i < 5; i++) {
			u.treat();
		}
		if (u.getState() == UnitState.TREATING)
			u.treat();
		assertEquals(
				"Ambulance should increment stepsPerCycle by 2 after it is done with its mission",
				oldStepsPerCycle + 2, u.getStepsPerCycle());

		// ************************************************//
		// Unit 2: GasControlUnit

		s = new Simulator(sos);
		GasControlUnit u1 = new GasControlUnit("GCU1", new Address(3, 4), 3,
				null);
		oldStepsPerCycle = u1.getStepsPerCycle();
		u.setWorldListener(s);
		ResidentialBuilding c11 = new ResidentialBuilding(new Address(3, 9));
		Disaster d1 = new GasLeak(3, c11);
		dshelper(d1);
		c11.setGasLevel(50);

		unitRespond(u1, c11, 5);

		u1.setState(UnitState.TREATING);

		for (int i = 0; i < 5; i++) {
			u1.treat();
		}
		if (u1.getState() == UnitState.TREATING)
			u1.treat();
		assertEquals(
				"GasControlUnit should increment stepsPerCycle by 2 after it is done with its mission",
				oldStepsPerCycle + 2, u1.getStepsPerCycle());
		// ************************************************//
		// Unit 3: Evacuator

		s = new Simulator(sos);
		Address ad = getAddressFromWorld(s, 3, 9);
		Evacuator u2 = new Evacuator("evacuator1",
				getAddressFromWorld(s, 3, 4), 5, null, 2);
		u2.setWorldListener(s);
		oldStepsPerCycle = u2.getStepsPerCycle();

		ResidentialBuilding b = new ResidentialBuilding(ad);
		b.setFoundationDamage(10);
		ArrayList<Citizen> citizensToBeTested = new ArrayList<Citizen>();
		for (int i = 1; i <= 2; i++) {
			Citizen c = new Citizen(ad, i + "", "citizen" + i, 15 + i, null);
			c.setWorldListener(s);
			citizensToBeTested.add(c);
			b.getOccupants().add(c);
		}
		Disaster d2 = new Collapse(3, b);
		dshelper(d2);

		unitRespond(u2, b, 5);

		for (int i = 0; i < 10; i++) {
			u2.cycleStep();
		}
		if (u2.getState() == UnitState.TREATING)
			u2.cycleStep();
		assertEquals(
				"Evacuator should increment stepsPerCycle by 2 after it is done with its mission",
				oldStepsPerCycle + 2, u2.getStepsPerCycle());

	}

	@Test(timeout = 3000)
	public void testUnitsStepsPerCycleDoesNotExceedMax() throws Exception {

		// Unit 1: Ambulance
		Simulator s = new Simulator(sos);
		Ambulance u = new Ambulance("ambulance1", new Address(3, 4), 19, null);
		u.setWorldListener(s);
		Citizen c1 = new Citizen(new Address(3, 9), "1", "citizen1", 15, null);
		Disaster d = new Injury(3, c1);
		dshelper(d);
		c1.setBloodLoss(50);
		unitRespond(u, c1, 5);

		u.setState(UnitState.TREATING);
		for (int i = 0; i < 5; i++) {
			u.treat();
		}
		if (u.getState() == UnitState.TREATING)
			u.treat();
		assertEquals("Ambulance stepsPerCycle should not exceed 20", 20,
				u.getStepsPerCycle());

		// *****************************************
		// Unit 2: GasControlUnit
		s = new Simulator(sos);
		GasControlUnit u1 = new GasControlUnit("GCU1", new Address(3, 4), 19,
				null);
		u1.setWorldListener(s);
		ResidentialBuilding c11 = new ResidentialBuilding(new Address(3, 9));
		Disaster d1 = new GasLeak(3, c11);
		dshelper(d1);
		c11.setGasLevel(50);

		unitRespond(u1, c11, 5);
		u1.setState(UnitState.TREATING);
		for (int i = 0; i < 5; i++) {
			u1.treat();
		}
		if (u1.getState() == UnitState.TREATING)
			u1.treat();
		assertEquals("GasControlUnit stepsPerCycle should not exceed 20", 20,
				u1.getStepsPerCycle());

		// ******************************************
		// Unit 3: Evacuator
		s = new Simulator(sos);
		Address ad = getAddressFromWorld(s, 3, 9);
		Evacuator u2 = new Evacuator("evacuator1",
				getAddressFromWorld(s, 3, 4), 19, null, 2);
		u2.setWorldListener(s);

		ResidentialBuilding b = new ResidentialBuilding(ad);
		b.setFoundationDamage(10);
		ArrayList<Citizen> citizensToBeTested = new ArrayList<Citizen>();
		for (int i = 1; i <= 2; i++) {
			Citizen c = new Citizen(ad, i + "", "citizen" + i, 15 + i, null);
			c.setWorldListener(s);
			citizensToBeTested.add(c);
			b.getOccupants().add(c);
		}
		Disaster d2 = new Collapse(3, b);
		dshelper(d2);

		unitRespond(u2, b, 5);
		for (int i = 0; i < 10; i++) {
			u2.cycleStep();
		}
		if (u2.getState() == UnitState.TREATING)
			u2.cycleStep();
		assertEquals("Evacuator stepsPerCycle should not exceed 20", 20,
				u2.getStepsPerCycle());

	}

	@Test(timeout = 3000)
	public void testFireTruckIncreasesStepsPerCycleAfterMission()
			throws Exception {
		Simulator s = new Simulator(sos);
		FireTruck u = new FireTruck("firetruck1", new Address(3, 4), 3, null);
		u.setWorldListener(s);
		int oldStepsPerCycle = u.getStepsPerCycle();
		ResidentialBuilding c1 = new ResidentialBuilding(new Address(3, 9));
		Disaster d = new Fire(3, c1);
		dshelper(d);
		c1.setFireDamage(50);

		unitRespond(u, c1, 5);

		u.setState(UnitState.TREATING);
		for (int i = 0; i < 5; i++) {
			u.treat();
		
		}
		if (u.getState() == UnitState.TREATING)
			u.treat();
		
		assertEquals(
				"FireTruck should increment stepsPerCycle by 3 after it is done with its mission",
				oldStepsPerCycle + 3, u.getStepsPerCycle());

	}
	
	@Test(timeout = 3000)
	public void testFireTruckStepsPerCycleDoesNotExceedMax() throws Exception {
		Simulator s = new Simulator(sos);
		FireTruck u = new FireTruck("firetruck1", new Address(3, 4), 19, null);
		u.setWorldListener(s);
		ResidentialBuilding c1 = new ResidentialBuilding(new Address(3, 9));
		Disaster d = new Fire(3, c1);
		dshelper(d);
		c1.setFireDamage(50);

		unitRespond(u, c1, 5);

		u.setState(UnitState.TREATING);
		for (int i = 0; i < 5; i++) {
			u.treat();

		}
		if (u.getState() == UnitState.TREATING)
			u.treat();
			assertEquals("FireTruck stepsPerCycle should not exceed 20", 20,
					u.getStepsPerCycle());

	}

	// HELPERS
	private void testGetterLogic(Object createdObject, String name, Object value)
			throws Exception {
		Field f = null;
		Class curr = createdObject.getClass();
		while (f == null) {
			if (curr == Object.class)
				fail("Class " + createdObject.getClass().getSimpleName()
						+ " should have the instance variable \"" + name
						+ "\".");
			try {
				f = curr.getDeclaredField(name);
			} catch (NoSuchFieldException e) {
				curr = curr.getSuperclass();
			}
		}
		f.setAccessible(true);
		f.set(createdObject, value);
		Character c = name.charAt(0);
		String methodName = "get" + Character.toUpperCase(c)
				+ name.substring(1, name.length());
		if (value.getClass().equals(Boolean.class))
			methodName = "is" + Character.toUpperCase(c)
					+ name.substring(1, name.length());
		Method m = createdObject.getClass().getMethod(methodName);
		assertEquals("The method \"" + methodName + "\" in class "
				+ createdObject.getClass().getSimpleName()
				+ " should return the correct value of variable \"" + name
				+ "\".", value, m.invoke(createdObject));
	}

	private void testConstructorInitialization(Object createdObject,
			String[] names, Object[] values) throws NoSuchMethodException,
			SecurityException, IllegalArgumentException, IllegalAccessException {
		for (int i = 0; i < names.length; i++) {
			Field f = null;
			Class curr = createdObject.getClass();
			String currName = names[i];
			Object currValue = values[i];
			while (f == null) {
				if (curr == Object.class)
					fail("Class " + createdObject.getClass().getSimpleName()
							+ " should have the instance variable \""
							+ currName + "\".");
				try {
					f = curr.getDeclaredField(currName);
				} catch (NoSuchFieldException e) {
					curr = curr.getSuperclass();
				}
			}
			f.setAccessible(true);
			assertEquals("The constructor of the "
					+ createdObject.getClass().getSimpleName()
					+ " class should initialize the instance variable \""
					+ currName + "\" correctly.", currValue,
					f.get(createdObject));
		}
	}

	private void testGetterMethodExistsInClass(Class aClass, String methodName,
			Class returnedType, boolean writeVariable) {
		Method m = null;
		boolean found = true;
		try {
			m = aClass.getDeclaredMethod(methodName);
		} catch (Exception e) {
			found = false;
		}
		String varName = "";
		if (returnedType == boolean.class)
			varName = methodName.substring(2).toLowerCase();
		else
			varName = methodName.substring(3).toLowerCase();
		if (writeVariable) {
			assertTrue("The \"" + varName + "\" instance variable in class "
					+ aClass.getSimpleName() + " is a READ variable.", found);
			assertTrue("Incorrect return type for " + methodName
					+ " method in " + aClass.getSimpleName() + " class.", m
					.getReturnType().isAssignableFrom(returnedType));
		} else {
			assertFalse("The \"" + varName + "\" instance variable in class "
					+ aClass.getSimpleName() + " is not a READ variable.",
					found);
		}
	}

	private void testSetterMethodExistsInClass(Class aClass, String methodName,
			Class inputType, boolean writeVariable) {
		Method[] methods = aClass.getDeclaredMethods();
		String varName = methodName.substring(3).toLowerCase();
		if (writeVariable) {
			assertTrue("The \"" + varName + "\" instance variable in class "
					+ aClass.getSimpleName() + " is a WRITE variable.",
					containsMethodName(methods, methodName));
		} else {
			assertFalse("The \"" + varName + "\" instance variable in class "
					+ aClass.getSimpleName() + " is not a WRITE variable.",
					containsMethodName(methods, methodName));
			return;
		}
		Method m = null;
		boolean found = true;
		try {
			m = aClass.getDeclaredMethod(methodName, inputType);
		} catch (NoSuchMethodException e) {
			found = false;
		}
		assertTrue(aClass.getSimpleName() + " class should have " + methodName
				+ " method that takes one " + inputType.getSimpleName()
				+ " parameter.", found);
		assertTrue("Incorrect return type for " + methodName + " method in "
				+ aClass.getSimpleName() + ".",
				m.getReturnType().equals(Void.TYPE));
	}

	private static boolean containsMethodName(Method[] methods, String name) {
		for (Method method : methods) {
			if (method.getName().equals(name))
				return true;
		}
		return false;
	}

	private void testInstanceVariableIsPresent(Class aClass, String varName,
			boolean implementedVar) throws SecurityException {
		boolean thrown = false;
		try {
			aClass.getDeclaredField(varName);
		} catch (NoSuchFieldException e) {
			thrown = true;
		}
		if (implementedVar) {
			assertFalse("There should be \"" + varName
					+ "\" instance variable in class " + aClass.getSimpleName()
					+ ".", thrown);
		} else {
			assertTrue(
					"The instance variable \"" + varName
							+ "\" should not be declared in class "
							+ aClass.getSimpleName() + ".", thrown);
		}
	}

	private void testInstanceVariableIsPrivate(Class aClass, String varName)
			throws NoSuchFieldException, SecurityException {
		Field f = aClass.getDeclaredField(varName);
		assertEquals("The \"" + varName + "\" instance variable in class "
				+ aClass.getSimpleName()
				+ " should not be accessed outside that class.", 2,
				f.getModifiers());
	}

	private static void unitRespond(Unit u, Rescuable r, int dist)
			throws IllegalArgumentException, IllegalAccessException {
		if (r != null && u.getState() == UnitState.TREATING) {

			Disaster curr = r.getDisaster();
			curr.setActive(true);
		}

		Field targetField = null;
		Class curr0 = u.getClass();
		while (targetField == null) {
			if (curr0 == Object.class)
				fail("Class " + u.getClass().getSimpleName()
						+ " should have the instance variable \"" + "disaster"
						+ "\".");
			try {
				targetField = curr0.getDeclaredField("target");
			} catch (NoSuchFieldException e) {
				curr0 = curr0.getSuperclass();
			}
		}
		targetField.setAccessible(true);
		targetField.set(u, r);

		u.setState(UnitState.RESPONDING);
		int distanceToTarget = dist;

		Field f = null;
		Class curr = u.getClass();
		while (f == null) {
			if (curr == Object.class)
				fail("Class " + u.getClass().getSimpleName()
						+ " should have the instance variable \"" + "disaster"
						+ "\".");
			try {
				f = curr.getDeclaredField("distanceToTarget");
			} catch (NoSuchFieldException e) {
				curr = curr.getSuperclass();
			}
		}
		f.setAccessible(true);
		f.set(u, distanceToTarget);

	}

	private static void dshelper(Disaster d) throws IllegalArgumentException,
			IllegalAccessException {
		Rescuable target = d.getTarget();
		if (target instanceof Citizen) {
			Citizen c = (Citizen) target;
			Field f = null;
			Class curr = c.getClass();
			while (f == null) {
				if (curr == Object.class)
					fail("Class " + c.getClass().getSimpleName()
							+ " should have the instance variable \""
							+ "disaster" + "\".");
				try {
					f = curr.getDeclaredField("disaster");
				} catch (NoSuchFieldException e) {
					curr = curr.getSuperclass();
				}
			}
			f.setAccessible(true);
			f.set(c, d);
		} else {
			ResidentialBuilding b = (ResidentialBuilding) target;
			Field f = null;
			Class curr = b.getClass();
			while (f == null) {
				if (curr == Object.class)
					fail("Class " + b.getClass().getSimpleName()
							+ " should have the instance variable \""
							+ "disaster" + "\".");
				try {
					f = curr.getDeclaredField("disaster");
				} catch (NoSuchFieldException e) {
					curr = curr.getSuperclass();
				}
			}
			f.setAccessible(true);
			f.set(b, d);
		}
		d.setActive(true);
	}

	private static Address getAddressFromWorld(Simulator s, int x, int y)
			throws IllegalArgumentException, IllegalAccessException {

		Field f = null;
		Class curr = s.getClass();
		while (f == null) {
			if (curr == Object.class)
				fail("Class " + s.getClass().getSimpleName()
						+ " should have the instance variable \"" + "world"
						+ "\".");
			try {
				f = curr.getDeclaredField("world");
			} catch (NoSuchFieldException e) {
				curr = curr.getSuperclass();
			}
		}
		f.setAccessible(true);
		Address myWorld[][] = (Address[][]) f.get(s);
		return myWorld[x][y];
	}

	public class MyDisaster extends Disaster {

		public MyDisaster(int i, Citizen c1) {
			super(i, c1);
		}

		@Override
		public void cycleStep() {

		}

	}

	class MyMedicalUnit extends MedicalUnit {

		public MyMedicalUnit(String unitID, Address location, int stepsPerCycle) {
			super(unitID, location, stepsPerCycle, null);
		}

		@Override
		public void treat() {

		}

		public void cycleStep() {

		}

	}
}
