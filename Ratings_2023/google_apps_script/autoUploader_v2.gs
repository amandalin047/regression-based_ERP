function applyLabelandUpload() {

   const threads = GmailApp.getInboxThreads(0,15);
   threads.forEach(threadValue1);
}

function threadValue1(thrd) {
  const amanda = "amanda.lin0103@gmail.com";

  const [upload_subject_1, upload_subject_2, fail_subject] = ["RE: Submission Received", "RE: Submission Received (Further Confirmation Needed)","RE: Subission Failed"];
  const upload_body_1 = "We've received your [] submission!";
  const upload_body_2 = "We've received your {} submission! However, we coudn't find your <> email address in our [] Google form...";

  const [a, d, f] = [GmailApp.getUserLabelByName("Uploaded"), GmailApp.getUserLabelByName("Pending"), GmailApp.getUserLabelByName("Failed")];

  const messages = thrd.getMessages();

  var [check, got, sendER, dontSEND] = ["", "", "", []];
  messages.forEach((messageValue)=>{
    
    try {
      var sender = messageValue.getFrom().split("<")[1].split(">")[0];
    }
    catch(err){
      var sender = messageValue.getFrom();
    }
    sendER = sender;

    const folder1 = DriveApp.getFolderById("ID1");
    const folder2_1 = DriveApp.getFolderById("ID2_1");
    const folder2_2 = DriveApp.getFolderById("ID2_2");
    const folder3 = DriveApp.getFolderById("ID3");
    const folder4 = DriveApp.getFolderById("ID4");

    const responses1 = SpreadsheetApp.openByUrl("URL1").getSheetByName("Sheet 1");
    const responses2_1 = SpreadsheetApp.openByUrl("URL2_1").getSheetByName("Sheet 1");
    const responses2_2 = SpreadsheetApp.openByUrl("URL2_2").getSheetByName("Sheet 1");
    const responses3 = SpreadsheetApp.openByUrl("URL3").getSheetByName("Sheet 1");
    const responses4 = SpreadsheetApp.openByUrl("URL4").getSheetByName("Sheet 1");
    
    var attach = messageValue.getAttachments();
   
    if(thrd.getLabels().includes(d)){
      Logger.log("pending");
      if (attach.length > 0){
        Logger.log("attachmet(s) found");

        attach.forEach((attachValue) => {
          Logger.log(attachValue.getName());
          if (attachValue.getName().includes("plaus")){
            
            dontSEND.push(false);

            folder1.createFile(attachValue.copyBlob()).setName(attachValue.getName()+sender);
            thrd.addLabel(a);
            if (!responses1.getRange("D:D").getValues().flat().includes(sender)){
              check += "[plausibility & expectancy] ";
              messageValue.star();
            } else {
              got += "[plausibility & expectancy] ";
              var vals = responses1.getRange("D:D").getValues().flat();
              var row = vals.indexOf(sender) + 1 ;
              var cell = responses1.getRange(row, 14);
              Logger.log(row);
              Logger.log(cell.getValue());
              cell.setValue("V");
            }
            
          } else if (attachValue.getName().includes("frame") || attachValue.getName().includes("sentence")){
            
            dontSEND.push(false);

            folder2_1.createFile(attachValue.copyBlob()).setName(attachValue.getName()+sender);
            thrd.addLabel(a);
            if (!responses2_1.getRange("D:D").getValues().flat().includes(sender)){
              check += "[sentence valence & arousal] ";
              messageValue.star();
            } else {
              got += "[sentence valence & arousal] ";
              var vals = responses2_1.getRange("D:D").getValues().flat();
              var row = vals.indexOf(sender) + 1 ;
              var cell = responses2_1.getRange(row, 14);
              Logger.log(row);
              cell.setValue("V");
              Logger.log(cell.getValue());
            }

          } else if (attachValue.getName().includes("WORD")){

            dontSEND.push(false);

            folder2_2.createFile(attachValue.copyBlob()).setName(attachValue.getName()+sender);
            thrd.addLabel(a);
            if (!responses2_2.getRange("D:D").getValues().flat().includes(sender)){
              check += "[WORD VALENCE & AROUSAL] ";
              messageValue.star();
            } else {
              got += "[WORD VALENCE & AROUSAL] ";
              var vals = responses2_2.getRange("D:D").getValues().flat();
              var row = vals.indexOf(sender) + 1 ;
              var cell = responses2_2.getRange(row, 14);
              Logger.log(row);
              cell.setValue("V");
              Logger.log(cell.getValue());
            }
            
          } else if (attachValue.getName().includes("fami")){

            dontSEND.push(false);

            folder3.createFile(attachValue.copyBlob()).setName(attachValue.getName()+sender);
            thrd.addLabel(a);
            if (!responses3.getRange("D:D").getValues().flat().includes(sender)){
              check += "[familiarity & concreteness] ";
              messageValue.star();
            } else {
              got += "[familiarity & concreteness] ";
              var vals = responses3.getRange("D:D").getValues().flat();
              var row = vals.indexOf(sender) + 1 ;
              var cell = responses3.getRange(row, 14);
              Logger.log(row);
              cell.setValue("V");
              Logger.log(cell.getValue());
            }
            
          } else if (attachValue.getName().includes("word valence")){
            
            dontSEND.push(false);

            folder4.createFile(attachValue.copyBlob()).setName(attachValue.getName()+sender);
            thrd.addLabel(a);
            if (!responses4.getRange("D:D").getValues().flat().includes(sender)){
              check += "[word valence & arousal] ";
              messageValue.star();
            } else {
              got += "[word valence & arousal] ";
              var vals = responses4.getRange("D:D").getValues().flat();
              var row = vals.indexOf(sender) + 1 ;
              var cell = responses4.getRange(row, 14);
              Logger.log(row);
              cell.setValue("V");
              Logger.log(cell.getValue());
            }
          } else {
            dontSEND.push(true);
            messageValue.star();
            thrd.addLabel(f);
            MailApp.sendEmail(amanda, fail_subject, sender);
          }
       })

      } else {
        dontSEND.push(true);
        messageValue.star();
        thrd.addLabel(f);
        MailApp.sendEmail(amanda, fail_subject, sender);
      }

    }
  })

if (thrd.getLabels().includes(a) && thrd.getLabels().includes(f)){
  thrd.removeLabel(f);
}

if (thrd.getLabels().includes(d)){
  if (check == "" && dontSEND.includes(false)) {
    Logger.log(check);
    try{
        MailApp.sendEmail(sendER, upload_subject_1, upload_body_1.split("[")[0] + got + upload_body_1.split("]")[1]);
        Logger.log("sent to "+sendER);
       }
    catch(err){
        messageValue.star();
        MailApp.sendEmail(amanda, upload_subject_1, upload_body_1.split("[")[0] + got + upload_body_1.split("]")[1]);
        Logger.log("sent to "+amanda);
       }
  } else if (dontSEND.includes(false)) {
    try{
       MailApp.sendEmail(sendER, upload_subject_2, upload_body_2.split("{")[0] + got + upload_body_2.split("}")[1].split("[")[0] + check + upload_body_2.split("}")[1].split("]")[1].split("<")[0] + sendER + upload_body_2.split("}")[1].split("]")[1].split(">")[1]);
       Logger.log("sent to "+sendER);
       }
    catch{
       MailApp.sendEmail(amanda, upload_subject_2, upload_body_2.split("{")[0] + got + upload_body_2.split("}")[1].split("[")[0] + check + upload_body_2.split("}")[1].split("]")[1].split("<")[0] + sendER + upload_body_2.split("}")[1].split("]")[1].split(">")[1]);
       Logger.log("sent to "+sendER);
      }
  } 

thrd.removeLabel(d);
Logger.log("removed label");
}
}
