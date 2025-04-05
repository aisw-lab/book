#include <iostream>
#include <fl/FuzzyLite.h>

using namespace fuzzylite;

int main() {
    // 1. 엔진 생성
    Engine* engine = new Engine("AirConditioner");

    // 2. 입력 변수 생성 및 설정
    InputVariable* temperature = new InputVariable("Temperature");
    temperature->setRange(15.0, 40.0);
    temperature->addTerm(new Triangle("Cold", 15.0, 15.0, 25.0));
    temperature->addTerm(new Triangle("Normal", 20.0, 25.0, 30.0));
    temperature->addTerm(new Triangle("Hot", 25.0, 35.0, 40.0));
    engine->addInputVariable(temperature);

    InputVariable* humidity = new InputVariable("Humidity");
    humidity->setRange(30.0, 90.0);
    humidity->addTerm(new Triangle("Dry", 30.0, 30.0, 60.0));
    humidity->addTerm(new Triangle("Normal", 40.0, 60.0, 80.0));
    humidity->addTerm(new Triangle("Humid", 60.0, 80.0, 90.0));
    engine->addInputVariable(humidity);

    // 3. 출력 변수 생성 및 설정
    OutputVariable* coolingPower = new OutputVariable("CoolingPower");
    coolingPower->setRange(0.0, 100.0);
    coolingPower->addTerm(new Triangle("Low", 0.0, 25.0, 50.0));
    coolingPower->addTerm(new Triangle("Medium", 25.0, 50.0, 75.0));
    coolingPower->addTerm(new Triangle("High", 50.0, 75.0, 100.0));

    // Defuzzifier 설정 (Centroid 또는 WeightedAverage)
    coolingPower->setDefuzzifier(new Centroid()); // 또는 new WeightedAverage()

    // Aggregation 연산자 설정 (Maximum 또는 AlgebraicSum)
    coolingPower->setAggregation(new Maximum()); // 또는 new AlgebraicSum()

    engine->addOutputVariable(coolingPower);

    // 4. 규칙 생성 및 설정
    RuleBlock* ruleBlock = new RuleBlock("RuleBlock1");

    // Conjunction 연산자 설정 (Minimum 또는 AlgebraicProduct)
    ruleBlock->setConjunction(new Minimum()); // 또는 new AlgebraicProduct()

    // Disjunction 연산자 설정 (Maximum 또는 AlgebraicSum)
    ruleBlock->setDisjunction(new Maximum()); // 또는 new AlgebraicSum()

    // Implication 연산자 설정 (Minimum 또는 AlgebraicProduct)
    ruleBlock->setImplication(new Minimum()); // 또는 new AlgebraicProduct()

    ruleBlock->addRule(Rule::parse("if Temperature is Cold and Humidity is Dry then CoolingPower is Low", engine));
    ruleBlock->addRule(Rule::parse("if Temperature is Normal and Humidity is Normal then CoolingPower is Medium", engine));
    ruleBlock->addRule(Rule::parse("if Temperature is Hot or Humidity is Humid then CoolingPower is High", engine));
    engine->addRuleBlock(ruleBlock);

    // 5. 엔진 정보 출력 (선택 사항)
    //std::cout << engine->toString() << std::endl;

    // 6. 입력 값 설정
    temperature->setValue(30.0); // 현재 온도
    humidity->setValue(70.0);   // 현재 습도

    // 7. 퍼지 추론 실행
    engine->process();

    // 8. 결과 출력
    std::cout << "현재 온도: " << temperature->getValue() << "°C" << std::endl;
    std::cout << "현재 습도: " << humidity->getValue() << "%" << std::endl;
    std::cout << "냉방 강도: " << coolingPower->getValue() << "%" << std::endl;

    // 9. 메모리 해제
    delete engine;

    return 0;
}
