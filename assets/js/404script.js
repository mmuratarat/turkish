        var errorMessages = ["I'll be back!",
                             'Hello, IT. Have you tried turning it off and on again?',
                             'Welcome to This Page. The first rule of This Page is: you do not talk about This Page.',
                             'Houston, we have a problem.',
                             "Error messages! Why'd it have to be error messages?!",
                             "I'm sorry, Dave. I'm afraid I can't do that.",
                             "You're gonna need a bigger server.",
                             "God help us; we're in the hands of engineers.",
                             "I'm as mad as hell, and I'm not going to take this anymore!",
                             "But we'll see each other soon. Won't we?",
                             "You're the wrong guy on the wrong page at the wrong time."
                            ];
      var item = errorMessages[Math.floor(Math.random() * errorMessages.length)];
      document.getElementById("error-message").innerHTML = "<p>" + item + "</p>";
